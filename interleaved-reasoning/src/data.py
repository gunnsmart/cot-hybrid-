"""Synthetic tasks A/B/C with *measured* vs *label* difficulty.

Design rules
------------
* Every task reduces to a chain of value updates, so the *only* thing the
  processor must do is decide, per stage, whether to keep the working value
  in the (lossy) latent channel or anchor it to the output trace.
* Two difficulty notions are defined per sample:
    - ``label``:    nominal complexity given by the surface form
                    (operators in the expression / facts given / instructions given).
    - ``measured``: complexity of the *minimal* computation that is actually
                    required (trivial ops like ``x*1`` or ``nop`` instructions
                    are eliminated by an oracle solver).
  The two coincide only when the surface form contains no redundancy, so the
  two Pearson correlations requested by the measurement rules can disagree.
* Gold intermediate traces are produced for analysis only. They are NEVER
  used as supervision (no external mode supervision is allowed).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Vocabularies (all fit in 64 tokens). Token 0 is reserved (<pad>).
# ---------------------------------------------------------------------------

# Task A -- arithmetic ------------------------------------------------------
A_DIGIT = {str(i): i for i in range(1, 10)}          # '1'..'9' -> 1..9
A_SYM = {"+": 10, "x": 11, "(": 12, ")": 13}
A_VAL0 = 14                                          # value v -> token 14+v
A_VMAX = 49                                          # values 0..49 fit in 14..63

# Task B -- logic -----------------------------------------------------------
B_ENT0 = 1                                           # entity e -> token 1+e (0..25)
B_ENTMAX = 25
B_ARROW, B_DOT, B_QUERY = 28, 29, 30

# Task C -- recall ----------------------------------------------------------
C_VAL0 = 1                                           # value v -> token 1+v (0..49)
C_VMAX = 49
C_START = 51
C_OPS = {"dbl": 52, "add1": 53, "add2": 54, "sub1": 55, "sub2": 56, "half": 57, "nop": 58}
C_TRIVIAL = {"nop"}

VOCAB = 64


@dataclass(frozen=True)
class Sample:
    tokens: tuple            # input token ids (no padding)
    answer: int              # gold answer token
    label: int               # nominal difficulty (surface complexity)
    measured: int            # minimal required computation
    trace: tuple = field(default_factory=tuple)  # gold intermediate tokens (analysis only)


# ---------------------------------------------------------------------------
# Task A: multi-step arithmetic, e.g. "(3+4)x(2+5)"
# ---------------------------------------------------------------------------

def _sample_tree(rng: random.Random, depth: int):
    """Random binary tree with `depth` internal nodes (each an operator)."""
    if depth == 1:
        return ("leaf", rng.randint(1, 9))
    left_depth = rng.randint(1, depth - 1)
    right_depth = depth - left_depth
    return (rng.choice(["+", "x"]),
            _sample_tree(rng, left_depth),
            _sample_tree(rng, right_depth))


def _eval_tree(node):
    """Return (value, [(op_or_None, value) ... internal values in postorder])."""
    if node[0] == "leaf":
        return node[1], []
    _, l, r = node
    lv, lints = _eval_tree(l)
    rv, rints = _eval_tree(r)
    val = lv + rv if node[0] == "+" else lv * rv
    return val, lints + rints + [(node[0], val)]


def gen_arithmetic(rng: random.Random, n: int,
                   p_trivial: float = 0.30,
                   depth_range=(2, 7)) -> list[Sample]:
    """Expressions over digits 1..9 with + and x, all values <= A_VMAX.

    label    = number of operators in the surface expression.
    measured = number of operators that are NOT trivially ``x 1``
               (oracle simplification: ``x*1`` needs no new intermediate value).
    """
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        depth = rng.randint(*depth_range)
        tree = _sample_tree(rng, depth)
        # Optionally force a trivial factor: replace one leaf under a 'x' node by 1.
        if rng.random() < p_trivial:
            tree = _force_trivial(rng, tree)
        val, internals = _eval_tree(tree)
        if any(v > A_VMAX for _, v in internals) or val > A_VMAX:
            continue
        expr = _to_infix(tree)
        tokens = []
        for ch in expr:
            if ch.isdigit():
                tokens.append(A_DIGIT[ch])
            elif ch == "*":
                tokens.append(A_SYM["x"])
            else:
                tokens.append(A_SYM[ch])
        nops = sum(1 for op, _ in internals)
        nontriv = sum(1 for op, _ in internals if not (op == "x" and _ == 1))
        # measured = #intermediate values that must actually be produced
        measured = nontriv if nontriv >= 1 else 1
        out.append(Sample(tuple(tokens), A_VAL0 + val, nops, measured,
                          tuple(A_VAL0 + v for _, v in internals)))
    if len(out) < n:
        raise RuntimeError(f"arithmetic generator exhausted (got {len(out)}/{n})")
    return out


def _force_trivial(rng: random.Random, node):
    if node[0] == "leaf":
        return node
    op, l, r = node
    if op == "x" and rng.random() < 0.7:
        if rng.random() < 0.5:
            return ("x", ("leaf", 1), _force_trivial(rng, r))
        return ("x", _force_trivial(rng, l), ("leaf", 1))
    return (op, _force_trivial(rng, l), _force_trivial(rng, r))


def _to_infix(node) -> str:
    if node[0] == "leaf":
        return str(node[1])
    _, l, r = node
    return f"({_to_infix(l)}{node[0]}{_to_infix(r)})"


# ---------------------------------------------------------------------------
# Task B: multi-hop logical chains, e.g. "A->B. B->C. C->D. Q A?"
# ---------------------------------------------------------------------------

def gen_logic(rng: random.Random, n: int,
              hops_range=(2, 8), max_distractors: int = 2) -> list[Sample]:
    """Facts e_i -> e_{i+1} plus distractor facts.

    label    = number of facts given (chain edges + distractors).
    measured = number of hops actually needed to answer the query
               (chain length; distractors are redundant).
    """
    out = []
    tries = 0
    while len(out) < n and tries < n * 100:
        tries += 1
        hops = rng.randint(*hops_range)
        ents = rng.sample(range(B_ENTMAX + 1), hops + 1)
        facts = list(zip(ents, ents[1:]))
        ndist = rng.randint(0, max_distractors) if rng.random() < 0.6 else 0
        used = set(ents)
        for _ in range(ndist):
            choice = rng.random()
            if choice < 0.5:
                # redundant: repeat an existing chain edge
                facts.append(rng.choice(facts))
            else:
                # side branch to a fresh entity (dead end)
                if len(used) >= B_ENTMAX + 1:
                    break
                side = rng.choice([e for e in range(B_ENTMAX + 1) if e not in used])
                used.add(side)
                facts.append((rng.choice(ents), side))
        rng.shuffle(facts)
        label = len(facts)
        measured = hops
        tokens = []
        for a, b in facts:
            tokens += [B_ENT0 + a, B_ARROW, B_ENT0 + b, B_DOT]
        tokens += [B_QUERY, B_ENT0 + ents[0]]
        trace = tuple(B_ENT0 + e for e in ents[1:-1])
        out.append(Sample(tuple(tokens), B_ENT0 + ents[-1], label, measured, trace))
    if len(out) < n:
        raise RuntimeError(f"logic generator exhausted (got {len(out)}/{n})")
    return out


# ---------------------------------------------------------------------------
# Task C: compositional recall with checkpoints
# ---------------------------------------------------------------------------

def gen_recall(rng: random.Random, n: int,
               steps_range=(2, 8), p_nop: float = 0.35) -> list[Sample]:
    """Instruction programs: start with v; op1 op2 ... opD.

    label    = number of instructions given.
    measured = number of NON-nop instructions (the minimal required work).
    """
    out = []
    tries = 0
    while len(out) < n and tries < n * 100:
        tries += 1
        D = rng.randint(*steps_range)
        v = rng.randint(1, 9)
        ops = []
        ok = True
        for _ in range(D):
            if rng.random() < p_nop:
                ops.append("nop")
                continue
            choices = ["dbl", "add1", "add2", "sub1", "sub2"]
            if v % 2 == 0 and v >= 2:
                choices.append("half")
            op = rng.choice(choices)
            if op == "dbl":
                v2 = v * 2
            elif op == "add1":
                v2 = v + 1
            elif op == "add2":
                v2 = v + 2
            elif op == "sub1":
                v2 = v - 1
            elif op == "sub2":
                v2 = v - 2
            else:
                v2 = v // 2
            if v2 < 0 or v2 > C_VMAX:
                ok = False
                break
            ops.append(op)
            v = v2
        if not ok:
            continue
        measured = max(1, sum(1 for o in ops if o not in C_TRIVIAL))
        tokens = [C_START, C_VAL0 + rng.randint(1, 9)]
        # start value must be consistent: regenerate ops from that start
        start = tokens[1] - C_VAL0
        v = start
        for op in ops:
            if op == "dbl":
                v *= 2
            elif op == "add1":
                v += 1
            elif op == "add2":
                v += 2
            elif op == "sub1":
                v -= 1
            elif op == "sub2":
                v -= 2
            elif op == "half":
                v //= 2
        if v < 0 or v > C_VMAX:
            continue
        tokens += [C_OPS[o] for o in ops]
        out.append(Sample(tuple(tokens), C_VAL0 + v, D, measured))
    if len(out) < n:
        raise RuntimeError(f"recall generator exhausted (got {len(out)}/{n})")
    return out


GEN = {
    "arithmetic": gen_arithmetic,
    "logic": gen_logic,
    "recall": gen_recall,
}


# ---------------------------------------------------------------------------
# Datasets (deterministic, no disk I/O)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    task: str
    split: str
    samples: list
    max_len: int

    def __len__(self):
        return len(self.samples)

    def get(self, indices):
        """Collate `indices` into padded tensors."""
        xs = [self.samples[i] for i in indices]
        L = max(len(s.tokens) for s in xs)
        X = torch.full((len(xs), L), 0, dtype=torch.long)
        for r, s in enumerate(xs):
            X[r, : len(s.tokens)] = torch.tensor(s.tokens)
        y = torch.tensor([s.answer for s in xs], dtype=torch.long)
        label = torch.tensor([s.label for s in xs], dtype=torch.long)
        measured = torch.tensor([s.measured for s in xs], dtype=torch.long)
        return X, y, label, measured


def make_dataset(task: str, split: str, n: int, seed: int = 1234) -> Dataset:
    rng = random.Random(seed * 1000 + {"train": 1, "dev": 2, "test": 3}[split])
    samples = GEN[task](rng, n)
    return Dataset(task, split, samples, max(len(s.tokens) for s in samples))


def collate(samples, max_len: int):
    X = torch.full((len(samples), max_len), 0, dtype=torch.long)
    for r, s in enumerate(samples):
        X[r, : len(s.tokens)] = torch.tensor(s.tokens)
    y = torch.tensor([s.answer for s in samples], dtype=torch.long)
    return X, y
