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
                 q_conf: float = 0.35, t_conf: float = 0.1,
                 content: str = "argmax", mode_bottleneck: int = 0,
                 gate_ema: float = 0.0, reader_mode: str = "full",
                 schedule_every: int | None = None):
        super().__init__()
        assert arch in STAGES, arch
        self.d, self.n_stages, self.vocab, self.arch, self.sigma = d, n_stages, vocab, arch, sigma
        self.mechanism = mechanism
        self.reader_layers = reader_layers
        # reader_mode (PHASE 2, H2): "full" = the original reader sees the
        # whole input before stage 0; "gradual" = the input is split into
        # n_stages+1 chunks -- chunk 0 initializes h_0 and chunk l+1 is
        # injected into the state AFTER stage l through a per-stage
        # zero-initialized projection, so no stage ever sees the full input.
        assert reader_mode in ("full", "gradual")
        self.reader_mode = reader_mode
        # schedule_every (PHASE 2, H3): with force="schedule" the model emits
        # unconditionally at every k-th stage ((l+1) % k == 0) and NEVER
        # consults the confidence gate -- a fixed-position baseline for the
        # learned controller. None = not a schedule run.
        if schedule_every is not None:
            assert schedule_every >= 1
        self.schedule_every = schedule_every
        # Content gate: emit only when the readout is confident (maxp >= q_conf).
        # Soft: g = sigmoid((maxp - q_conf)/t_conf); hard: maxp >= q_conf.
        # q_conf <= 0 disables the gate (backward-compatible with early runs).
        self.q_conf, self.t_conf = q_conf, t_conf
        # gate_ema in [0,1): exponential memory on the per-stage content
        # confidence maxp BEFORE the gate is applied (soft AND hard paths).
        # 0 = instantaneous maxp (default, all prior runs). A positive value
        # smooths the stage-to-stage on/off spikiness of the gate, which is
        # input-specific and difficulty-unstructured (see calibration.md:
        # within-tier emit-count variance capped criterion 2).
        self.gate_ema = gate_ema
        # Soft-mode emission content: "argmax" (default) uses exactly the
        # hard-mode token, so the soft/hard gap is purely in the DECISION
        # (p*g vs the threshold) and never in the CONTENT; "expectation" is
        # the reference sketch's full-softmax expectation (kept as ablation:
        # it is what the content mismatch comes from).
        assert content in ("argmax", "expectation")
        self.content = content
        self.embed = nn.Embedding(vocab, d)
        if reader_mode == "gradual":
            # Bottleneck reader (H2): replaces the GRU reader entirely.
            # Task interface (group `base`), exactly like the reader it
            # replaces; zero-init => at step 0 the model behaves like a
            # full-reader model that receives no input after h_0.
            self.reader = None
            self.reader_proj = None
            self.chunk_proj = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(n_stages + 1)])
            for cp in self.chunk_proj:
                nn.init.zeros_(cp.weight)
        else:
            self.chunk_proj = None
            if reader_layers == 1:
                self.reader = nn.GRUCell(d, d)
                self.reader_proj = None
            else:  # 2-layer reader (task interface only, not a stage)
                self.reader = nn.GRU(d, d * 2, num_layers=2)
                self.reader_proj = nn.Linear(d * 2, d)
        self.stages = nn.ModuleList([STAGES[arch](d) for _ in range(n_stages)])
        self.mode_norm = nn.LayerNorm(d) if mechanism else nn.Identity()
        # mode_bottleneck > 0: mode decision reads a low-dim projection of the
        # state (smoother, more input-invariant policy); 0 = direct head.
        self.mode_bottleneck = mode_bottleneck
        if mechanism:
            if mode_bottleneck > 0:
                self.mode_head = nn.Sequential(
                    nn.Linear(d, mode_bottleneck), nn.ReLU(), nn.Linear(mode_bottleneck, 2))
            else:
                self.mode_head = nn.Linear(d, 2)
            for m_ in self.mode_head.modules() if mode_bottleneck > 0 else [self.mode_head]:
                if isinstance(m_, nn.Linear):
                    nn.init.zeros_(m_.weight)
                    nn.init.zeros_(m_.bias)
        else:
            self.mode_head = None
        self.readout = nn.Linear(d, vocab)

    # -- parameter accounting ------------------------------------------------
    def param_groups(self):
        def n(m):
            if m is None:
                return 0
            return sum(p.numel() for p in m.parameters())
        stages = n(self.stages)
        mechanism = 0
        if self.mechanism:
            mechanism = n(self.mode_norm) + n(self.mode_head)
        # The gradual chunk projections (PHASE 2) are task interface, not
        # mechanism: they replace the input reader, which every baseline
        # needs too, so they live in `base` (mechanism overhead is measured
        # against stages and must not grow with the reader variant).
        base = n(self.embed) + n(self.reader) + n(self.readout) + n(self.chunk_proj)
        total = stages + mechanism + base
        return {"stages": stages, "mechanism": mechanism, "base": base,
                "total": total, "overhead_pct": 100.0 * mechanism / stages}

    # -- input encoding --------------------------------------------------------
    def _chunk_embeds(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_stages+1, d): mean embedding of each input chunk.

        The sequence of length L is split into K = n_stages+1 contiguous
        chunks with boundaries `torch.linspace(0, L, K+1).round().long()`;
        chunk k covers tokens [b_k, b_{k+1}). Within a chunk the embedding
        is the mean over non-<pad> positions (<pad> = token id 0); a chunk
        with no real tokens (empty span or all pads) is the zero vector.
        """
        B, L = x.shape
        K = self.n_stages + 1
        e = self.embed(x)                                    # (B, L, d)
        pad = x == 0                                         # (B, L)
        bounds = torch.linspace(0, L, K + 1).round().long()  # (K+1,)
        chunks = []
        for k in range(K):
            a, b = int(bounds[k]), int(bounds[k + 1])
            if b <= a:                                       # empty span
                chunks.append(e.new_zeros(B, self.d))
                continue
            seg = e[:, a:b]                                  # (B, w, d)
            keep = (~pad[:, a:b]).to(e.dtype).unsqueeze(-1)  # (B, w, 1)
            denom = keep.sum(dim=1).clamp(min=1.0)           # (B, 1)
            chunks.append((seg * keep).sum(dim=1) / denom)   # (B, d)
        return torch.stack(chunks, dim=1)                    # (B, K, d)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """h_0 from the input token sequence."""
        if self.reader_mode == "gradual":
            # only chunk 0 is available at stage 0; the rest arrives after
            # the corresponding stage (see forward)
            return self.chunk_proj[0](self._chunk_embeds(x)[:, 0])
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
        force:   None | 'emit' | 'latent' | 'schedule'
                 (baselines / forced-mode evaluation; 'schedule' emits
                 unconditionally every schedule_every-th stage, gate bypassed)
        semantics: 'additive' (spec) | 'reset' (reference-sketch blend)
        noise_gen: torch.Generator for channel noise (eval determinism)
        returns: (h_final, P (B,N), trace (B, L, 2): [stage, token])
        """
        if self.reader_mode == "gradual":
            chunk_e = self._chunk_embeds(x)                    # (B, N+1, d)
            h = self.chunk_proj[0](chunk_e[:, 0])
        else:
            chunk_e = None
            h = self.encode(x)
        B = h.size(0)
        # PHASE 2 (H3): the forced schedule bypasses the confidence gate
        # entirely -- the point is to isolate "learned WHEN" from the gate
        # mechanism, so scheduled emissions are unconditional.
        gated = self.q_conf > 0 and force != "schedule"
        P = []
        G = []
        # per-sample trace: token per stage, -1 = not emitted yet
        trace_tokens = h.new_full((B, self.n_stages), -1, dtype=torch.long)
        mp_smooth = None  # running EMA of per-stage maxp (gate input)

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
            elif force == "schedule":
                # fixed-position baseline: emit every k-th stage, no choice
                assert self.schedule_every, "force='schedule' needs schedule_every"
                if (l + 1) % self.schedule_every == 0:
                    p = h.new_ones(B)
                else:
                    p = h.new_zeros(B)
            else:
                z = self.mode_head(self.mode_norm(h))
                p = torch.sigmoid(z[:, 1] - z[:, 0])          # (B,)
            P.append(p)

            dist = F.softmax(self.readout(h), dim=-1)         # (B, vocab)
            maxp = dist.max(-1).values                        # (B,)
            t_star = dist.argmax(-1)                          # (B,)
            if self.gate_ema > 0.0:
                mp_smooth = maxp if mp_smooth is None else \
                    self.gate_ema * mp_smooth + (1.0 - self.gate_ema) * maxp
                mp_g = mp_smooth
            else:
                mp_g = maxp
            if self.content == "expectation":
                e = dist @ self.embed.weight                  # sketch's expectation
            else:
                e = self.embed(t_star)                        # hard-mode content

            if mode == "soft":
                if gated:
                    g = torch.sigmoid((mp_g - self.q_conf) / self.t_conf)
                else:
                    g = torch.ones_like(maxp)
                G.append(g)
                if semantics == "additive":
                    h = h + (p * g).unsqueeze(-1) * e
                else:  # reset: the reference sketch's blend (1-p) h + p e
                    h = (1.0 - (p * g).unsqueeze(-1)) * h + (p * g).unsqueeze(-1) * e
            else:
                m = p >= tau
                if gated:
                    m = m & (mp_g >= self.q_conf)
                G.append(m.to(h.dtype))
                if m.any():
                    t = dist.argmax(-1)                       # (B,)
                    te = self.embed(t)
                    if semantics == "additive":
                        h = h + m.unsqueeze(-1) * te
                    else:
                        h = torch.where(m.unsqueeze(-1), te, h)
                    trace_tokens[m, l] = t[m]

            if chunk_e is not None:
                # gradual reader (H2): chunk l+1 joins the state only AFTER
                # stage l has run -- no stage ever sees the full input.
                h = h + self.chunk_proj[l + 1](chunk_e[:, l + 1])

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
