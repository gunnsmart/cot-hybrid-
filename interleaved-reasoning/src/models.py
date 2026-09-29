"""Stages, mode controller and the InterleavedProcessor.

Semantics (see PROBLEM.md for the formal definition):

    h_0 -> h_1 -> ... -> h_N,  h_l = stage_l(h_{l-1})  always computed

    latent mode:  h_l stays the stage output (silent)
    emit mode:    t_l = argmax readout(h_l);  h_l <- h_l + embed(t_l)   (additive,
                  the reference problem statement:  h_l = embed(t_l) + stage_l(h_{l-1}))

Soft training relaxes the hard decision with p_emit in [0,1] (sigmoid of the
mode-head logit difference) and the *expectation* of the emitted embedding:

    h' = h + p * E_{t ~ softmax(readout(h))}[ embed(t) ]

which is exactly the expectation of the additive hard rule -- fully
differentiable, deterministic, no sampling. A second relaxation,
``reset`` (the blend used in the reference sketch), is available for the
semantics ablation.

Channel noise: after every stage the latent state passes through
    h <- h + sigma * N(0, I)
identically at train and inference time ("same semantics"). This models the
imperfect retention of the latent channel and is the physical reason emitting
an anchor can be worth its price. sigma is a fixed channel property, NOT a
curriculum knob.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import VOCAB


# ---------------------------------------------------------------------------
# Stage architectures (the "underlying stage architecture" -- never modified
# by the mode mechanism; they only see the state h, plus for the transformer
# the trace so far, which is the only sequence available in a depth processor)
# ---------------------------------------------------------------------------

class StageMLP(nn.Module):
    """Reference Stage: h + Linear(GELU(LayerNorm(h)))."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, h):
        return h + self.mlp(F.gelu(self.norm(h)))


class StageGRU(nn.Module):
    """One GRU cell applied with the state as its own input (depth-as-time)."""

    def __init__(self, d: int):
        super().__init__()
        self.cell = nn.GRUCell(d, d)

    def forward(self, h):
        return self.cell(h, h)


class StageSSM(nn.Module):
    """Diagonal state-space cell (S4-style, one step, no external input):

        h' = a * h + (1 - a) * W h,   a = sigmoid(logit_a) in (0,1)^d

    i.e. a discretized linear recurrence whose input is a linear probe of the
    current state. a ~ 1 gives a pure latent filter; a ~ 0 a replacement.
    """

    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.W.weight, std=1.0 / math.sqrt(d))
        # start near identity: a -> 1
        self.logit_a = nn.Parameter(torch.full((d,), 3.0))

    def forward(self, h):
        a = torch.sigmoid(self.logit_a)
        return a * h + (1.0 - a) * self.W(h)


class StageTransformer(nn.Module):
    """Pre-LN single-head cross-attention of the state (query) over the
    emitted trace so far (keys/values), plus an MLP block. A learned <empty>
    token is always present as the first key, so samples with an empty trace
    attend to nothing informative (its value vector is learned ~0).
    """

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.empty = nn.Parameter(torch.zeros(d))
        for p in (self.q, self.k, self.v, self.o):
            nn.init.normal_(p.weight, std=1.0 / math.sqrt(d))

    def forward(self, h, trace_embeds, trace_mask):
        """trace_embeds: (B, L, d) (L may be 0); trace_mask: (B, L) bool."""
        B = h.size(0)
        e = self.empty.view(1, 1, -1).expand(B, 1, -1)
        if trace_embeds.size(1) == 0:
            kv = e
            mask = torch.ones(B, 1, dtype=torch.bool, device=h.device)
        else:
            kv = torch.cat([e, trace_embeds], dim=1)
            mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device),
                              trace_mask], dim=1)
        q = self.q(self.norm1(h)).unsqueeze(1)           # (B,1,d)
        k = self.k(kv).transpose(1, 2)                   # (B,d,L)
        a = (q @ k) / math.sqrt(h.size(-1))
        a = a.masked_fill(~mask.unsqueeze(1), float("-inf"))
        a = F.softmax(a, dim=-1)
        out = a @ self.v(kv)                             # (B,1,d)
        h = h + self.o(out.squeeze(1))
        h = h + self.fc2(F.gelu(self.fc1(self.norm2(h))))
        return h


STAGES = {
    "mlp": StageMLP,
    "gru": StageGRU,
    "ssm": StageSSM,
    "transformer": StageTransformer,
}


# ---------------------------------------------------------------------------
# The processor
# ---------------------------------------------------------------------------

class InterleavedProcessor(nn.Module):
    """Input reader -> N stages with a per-stage latent/emit mode controller.

    Parameter accounting (see PROBLEM.md criterion 6):
      * ``stages``      -- the base stage blocks (never touched by the mechanism)
      * ``mechanism``   -- mode_norm + mode_head  (what interleaved reasoning adds)
      * ``base``        -- shared embedding table, input reader, answer readout
        (task interface; present in every model including baselines)
    The embedding table is *shared* between input tokens and emitted trace
    tokens: both live in the same vocabulary (a value token is a value whether
    it is printed or computed).
    """

    def __init__(self, d: int, n_stages: int, vocab: int = VOCAB,
                 arch: str = "mlp", mechanism: bool = True,
                 sigma: float = 0.0, reader_layers: int = 1,
                 q_conf: float = 0.35, t_conf: float = 0.1):
        super().__init__()
        assert arch in STAGES, arch
        self.d, self.n_stages, self.vocab, self.arch, self.sigma = d, n_stages, vocab, arch, sigma
        self.mechanism = mechanism
        self.reader_layers = reader_layers
        # Content gate: emit only when the readout is confident (maxp >= q_conf).
        # Soft: g = sigmoid((maxp - q_conf)/t_conf); hard: maxp >= q_conf.
        # q_conf <= 0 disables the gate (backward-compatible with early runs).
        self.q_conf, self.t_conf = q_conf, t_conf
        self.embed = nn.Embedding(vocab, d)
        if reader_layers == 1:
            self.reader = nn.GRUCell(d, d)
            self.reader_proj = None
        else:  # 2-layer reader (task interface only, not a stage)
            self.reader = nn.GRU(d, d * 2, num_layers=2)
            self.reader_proj = nn.Linear(d * 2, d)
        self.stages = nn.ModuleList([STAGES[arch](d) for _ in range(n_stages)])
        self.mode_norm = nn.LayerNorm(d) if mechanism else nn.Identity()
        self.mode_head = nn.Linear(d, 2) if mechanism else None
        if mechanism:
            nn.init.zeros_(self.mode_head.weight)
            nn.init.zeros_(self.mode_head.bias)      # p_emit = 0.5 at init
        self.readout = nn.Linear(d, vocab)

    # -- parameter accounting ------------------------------------------------
    def param_groups(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())
        stages = n(self.stages)
        mechanism = 0
        if self.mechanism:
            mechanism = n(self.mode_norm) + n(self.mode_head)
        base = n(self.embed) + n(self.reader) + n(self.readout)
        total = stages + mechanism + base
        return {"stages": stages, "mechanism": mechanism, "base": base,
                "total": total, "overhead_pct": 100.0 * mechanism / stages}

    # -- input encoding --------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """h_0 from the input token sequence (depth-as-time GRU read)."""
        if self.reader_layers == 1:
            h = self.embed(x[:, 0])
            for t in range(1, x.size(1)):
                h = self.reader(self.embed(x[:, t]), h)
            return h
        e = self.embed(x).transpose(0, 1)        # (T, B, d)
        out, _ = self.reader(e)
        return self.reader_proj(out[-1])         # (B, d)

    # -- forward ----------------------------------------------------------------
    def forward(self, x, mode="soft", tau=0.5, force=None,
                semantics="additive", noise_gen=None):
        """
        mode:    'soft' (training) | 'hard' (inference)
        force:   None | 'emit' | 'latent'  (baselines / forced-mode evaluation)
        semantics: 'additive' (spec) | 'reset' (reference-sketch blend)
        noise_gen: torch.Generator for channel noise (eval determinism)
        returns: (h_final, P (B,N), trace (B, L, 2): [stage, token])
        """
        h = self.encode(x)
        B = h.size(0)
        P = []
        G = []
        # per-sample trace: token per stage, -1 = not emitted yet
        trace_tokens = h.new_full((B, self.n_stages), -1, dtype=torch.long)

        for l, stage in enumerate(self.stages):
            if self.arch == "transformer":
                prev = trace_tokens[:, :l]                    # emissions before l
                mask = prev >= 0
                if mask.any():
                    emb = self.embed(prev.clamp(min=0))       # (B,l,d), zeros at -1
                    h = stage(h, emb, mask)
                else:
                    h = stage(h, h.new_zeros(B, 0, self.d),
                              torch.zeros(B, 0, dtype=torch.bool, device=h.device))
            else:
                h = stage(h)

            if force == "emit":
                p = h.new_ones(B)
            elif force == "latent":
                p = h.new_zeros(B)
            else:
                z = self.mode_head(self.mode_norm(h))
                p = torch.sigmoid(z[:, 1] - z[:, 0])          # (B,)
            P.append(p)

            dist = F.softmax(self.readout(h), dim=-1)         # (B, vocab)
            maxp = dist.max(-1).values                        # (B,)
            e = dist @ self.embed.weight                      # (B, d) expected embed

            if mode == "soft":
                if self.q_conf > 0:
                    g = torch.sigmoid((maxp - self.q_conf) / self.t_conf)
                else:
                    g = torch.ones_like(maxp)
                G.append(g)
                if semantics == "additive":
                    h = h + (p * g).unsqueeze(-1) * e
                else:  # reset: the reference sketch's blend (1-p) h + p e
                    h = (1.0 - (p * g).unsqueeze(-1)) * h + (p * g).unsqueeze(-1) * e
            else:
                m = p >= tau
                if self.q_conf > 0:
                    m = m & (maxp >= self.q_conf)
                G.append(m.to(h.dtype))
                if m.any():
                    t = dist.argmax(-1)                       # (B,)
                    te = self.embed(t)
                    if semantics == "additive":
                        h = h + m.unsqueeze(-1) * te
                    else:
                        h = torch.where(m.unsqueeze(-1), te, h)
                    trace_tokens[m, l] = t[m]

            if self.sigma > 0:
                if noise_gen is not None:
                    eps = torch.randn(h.shape, device=h.device, generator=noise_gen)
                else:
                    eps = torch.randn_like(h)
                h = h + self.sigma * eps

        P = torch.stack(P, dim=1)                             # (B, N)
        G = torch.stack(G, dim=1)                             # (B, N) gate
        return h, P, trace_tokens, G

    def predict(self, x, mode="hard", tau=0.5, force=None,
                semantics="additive", noise_gen=None):
        h, P, trace, G = self.forward(x, mode=mode, tau=tau, force=force,
                                      semantics=semantics, noise_gen=noise_gen)
        return self.readout(h), P, trace, G
