#!/usr/bin/env python3
"""Render FINDINGS.md (and the README results block) FROM JSON ONLY.

Every number printed in FINDINGS.md is read from results/*.json at render
time; the prose lives in this script. Re-running this script must reproduce
the committed FINDINGS.md byte-for-byte (enforced by tests/test_findings.py).

Usage:  python -m scripts.render_findings
"""
from __future__ import annotations

import json
import os
import sys

from scripts import common

MAIN_RUNS = {"arithmetic": "A_mlp_main", "logic": "B_mlp_main", "recall": "C_mlp_main"}
ABLATIONS = {
    "A_abl_noprice":  "removing the emission price  (expect: Failure 3, token spam)",
    "A_abl_nocommit": "removing mode commitment     (expect: Failure 2, soft-hard gap)",
    "A_abl_nondeg":   "removing the non-degeneracy term (expect: Failure 1, mode collapse)",
    "A_abl_bare":     "removing ALL three regularizers (expect: all failures)",
    "A_abl_sig0":     "zero channel noise sigma=0   (expect: emission stops being useful)",
    "A_abl_reset":    "reset semantics instead of additive (informational variant)",
    "A_abl_redundant":"redundant surface p_trivial=0.4 (oracle 'measured' vs model-experienced difficulty)",
    "A_abl_nogate":  "confidence gate off (q_conf=0)  (expect: Failure 2 content gap -> c4 vs latent)",
}
SCALING = ["A_scale_n4", "A_scale_n8", "A_scale_n12", "A_scale_n24", "A_scale_n48"]
ARCH_A = {"A_gru": ("gru", "A_gru_emit", "A_gru_latent"),
          "A_ssm": ("ssm", "A_ssm_emit", "A_ssm_latent"),
          "A_transformer": ("transformer", "A_tf_emit", "A_tf_latent")}
ARCH_B = {"B_gru": ("gru", None, None), "B_transformer": ("transformer", None, None)}
# PHASE 2 pressure regimes (experiments/PHASE2.md): hybrid run -> (regime
# label, same-regime emit specialist, same-regime latent specialist).
# Schedule rows have no specialists; they are compared against the
# controller (A_mlp_main) in the hypothesis summary lines.
PHASE2_REGIMES = [
    ("A_s03", "sigma=0.3 (lossy channel)", "A_s03_emit", "A_s03_latent"),
    ("A_s05", "sigma=0.5 (lossy channel)", "A_s05_emit", "A_s05_latent"),
    ("A_bneck", "gradual input reader", "A_bneck_emit", "A_bneck_latent"),
    ("A_sched4", "forced schedule k=4 (~3 emits)", None, None),
    ("A_sched2", "forced schedule k=2 (6 emits)", None, None),
]
TOL = 1.05


def load(run_id):
    p = common.run_json_path(run_id)
    if not os.path.exists(p):
        return None
    j = common.read_json(p)
    return j if j.get("status") == "done" else None


def f(x, nd=3):
    return "pending" if x is None else f"{x:.{nd}f}"


def hard(j):
    return j["test"]["hard"] if (j and j.get("test")) else None


def ratios(j, emit_id, latent_id):
    h = hard(j)
    if h is None:
        return None, None
    je = load(emit_id)
    jl = load(latent_id)
    re_ = h["ce"] / hard(je)["ce"] if (je and hard(je)) else None
    rl = h["ce"] / hard(jl)["ce"] if (jl and hard(jl)) else None
    return re_, rl


def c1_pass(v):
    return v is not None and v > 0.1


def c2_pass(v):
    return v is not None and v > 0.3


def c3_pass(v):
    return v is not None and v < 0.1


def c4_pass(re_, rl):
    return re_ is not None and rl is not None and re_ <= TOL and rl <= TOL


def mark(b):
    return "PASS" if b else "FAIL" if b is False else "pending"


# ---------------------------------------------------------------------------

def sec_tldr(L):
    L.append("## TL;DR")
    L.append("")
    L.append("| task | acc (hard) | emits/input | c1 var(p) | c2 corr(measured) | c3 KL | c4 ratio emit/latent (tol 1.05) |")
    L.append("|---|---|---|---|---|---|---|")
    for task, rid in MAIN_RUNS.items():
        j = load(rid)
        h = hard(j)
        if h is None:
            L.append(f"| {task} | pending | | | | | |")
            continue
        re_, rl = ratios(j, rid.replace("_mlp_main", "_emit_base"),
                         rid.replace("_mlp_main", "_latent_base"))
        L.append(f"| {task} | {f(h['acc'])} | {f(h['mean_emits'], 2)} | {f(h['c1_var_p'])} "
                 f"| {f(h['c2_r_measured'])} | {f(h['c3_kl'])} "
                 f"| {f(re_, 3)}/{f(rl, 3)} ({mark(c4_pass(re_, rl))}) |")
    L.append("")


def sec_negative(L):
    L.append("## Negative results (first class)")
    L.append("")
    L.append("What broke when we removed a component, and whether the expected "
             "failure mode appeared. `confirmed` means the ablation reproduced the "
             "failure signature on the test set.")
    L.append("")
    L.append("| ablation | expected failure | observed (test) | verdict |")
    L.append("|---|---|---|---|")
    base = load("A_mlp_main")
    hb = hard(base) if base else None
    for rid, desc in ABLATIONS.items():
        j = load(rid)
        h = hard(j)
        if h is None:
            L.append(f"| {rid} | {desc} | pending | pending |")
            continue
        nb = j.get("negative_result")
        if rid == "A_abl_noprice":
            spam = h["mean_emits"] > 0.8 * j["config"]["n"]
            verdict = "confirmed" if spam else "refuted"
            obs = f"emits={f(h['mean_emits'], 2)}/{j['config']['n']} stages"
        elif rid == "A_abl_nocommit":
            verdict = "confirmed" if (not c3_pass(h["c3_kl"])) else "refuted"
            obs = f"c3 KL={f(h['c3_kl'])} (need <0.1)"
        elif rid == "A_abl_nondeg":
            verdict = "confirmed" if (not c1_pass(h["c1_var_p"])) else "refuted"
            obs = f"c1 var={f(h['c1_var_p'])} (need >0.1)"
        elif rid == "A_abl_bare":
            bad = sum([not c1_pass(h["c1_var_p"]), not c3_pass(h["c3_kl"]),
                       h["mean_emits"] > 0.8 * j["config"]["n"]])
            verdict = "confirmed" if bad >= 2 else "refuted"
            obs = (f"var={f(h['c1_var_p'])}, KL={f(h['c3_kl'])}, "
                   f"emits={f(h['mean_emits'], 2)}")
        elif rid == "A_abl_sig0":
            drop = hb is not None and h["c2_r_measured"] < 0.5 * max(0.3, hb["c2_r_measured"])
            verdict = "confirmed" if drop else "refuted"
            obs = f"c2 corr={f(h['c2_r_measured'])} (full recipe: {f(hb['c2_r_measured']) if hb else 'pending'})"
        else:  # reset
            verdict = "info"
            obs = f"acc={f(h['acc'])}, emits={f(h['mean_emits'], 2)}, KL={f(h['c3_kl'])}"
        if nb:
            obs += f" [{nb}]"
        L.append(f"| {rid} | {desc} | {obs} | {verdict} |")
    L.append("")
    L.append("Interpretation is written in `experiments/ABLATIONS.md` from the same JSON; "
             "nothing here is hand-edited after rendering.")
    L.append("")


def sec_criteria(L):
    L.append("## Success criteria scoreboard")
    L.append("")
    L.append(f"Tolerance for criterion 4 is **{TOL}** (explicit). c5 requires >= 2 "
             "architectures passing; c6 threshold is 5% of stage parameters.")
    L.append("")
    L.append("| run | arch | N | c1 | c2 measured | c2 label | c3 | c4 emit | c4 latent | c6 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for task, rid in MAIN_RUNS.items():
        rows.append((rid, "mlp",
                     f"{rid.replace('_mlp_main','_emit_base')}",
                     f"{rid.replace('_mlp_main','_latent_base')}"))
    for rid, (arch, eb, lb) in ARCH_A.items():
        rows.append((rid, arch, eb, lb))
    for rid, (arch, eb, lb) in ARCH_B.items():
        if eb:
            rows.append((rid, arch, eb, lb))
    n_pass_c5 = 0
    for rid, arch, eb, lb in rows:
        j = load(rid)
        h = hard(j)
        if h is None:
            L.append(f"| {rid} | {arch} | ? | pending |")
            continue
        pg = j["params"]
        re_, rl = (ratios(j, eb, lb) if eb and lb else (None, None))
        c1 = mark(c1_pass(h["c1_var_p"]))
        c2 = mark(c2_pass(h["c2_r_measured"]))
        c3 = mark(c3_pass(h["c3_kl"]))
        c4 = mark(c4_pass(re_, rl)) if (eb and lb) else "-"
        c6 = mark(pg["overhead_pct"] < 5.0)
        if c1 == "PASS" and c2 == "PASS" and c3 == "PASS" and c4 == "PASS" and c6 == "PASS":
            n_pass_c5 += 1
        L.append(f"| {rid} | {arch} | {j['config']['n']} | {f(h['c1_var_p'])} ({c1}) "
                 f"| {f(h['c2_r_measured'])} ({c2}) | {f(h['c2_r_label'])} "
                 f"| {f(h['c3_kl'])} ({c3}) | {f(re_, 3)} | {f(rl, 3)} "
                 f"| {f(pg['overhead_pct'], 2)}% ({c6}) |")
    L.append("")
    L.append(f"Criterion 5: **{n_pass_c5}** architecture rows pass all row criteria "
             f"(requirement: >= 2) -> {mark(n_pass_c5 >= 2)}.")
    L.append("")


def sec_scaling(L):
    L.append("## Scaling in N (task A, mlp, d=96)")
    L.append("")
    L.append("| N | acc | emits/input | c1 | c2 measured | c3 | p_eff(hard) |")
    L.append("|---|---|---|---|---|---|---|")
    for rid in SCALING:
        j = load(rid)
        h = hard(j)
        if h is None:
            L.append(f"| {rid[-2:]} | pending |")
            continue
        L.append(f"| {j['config']['n']} | {f(h['acc'])} | {f(h['mean_emits'], 2)} "
                 f"| {f(h['c1_var_p'])} | {f(h['c2_r_measured'])} | {f(h['c3_kl'])} "
                 f"| {f(h['p_eff'])} |")
    L.append("")


def sec_arch(L):
    L.append("## Architecture generalization")
    L.append("")
    L.append("Task A (with per-architecture baselines for c4) and task B (c1-c3 only):")
    L.append("")
    L.append("| run | arch | task | acc | c1 | c2 measured | c3 | c4 (emit/latent) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rid, (arch, eb, lb) in {**ARCH_A, **ARCH_B}.items():
        j = load(rid)
        h = hard(j)
        if h is None:
            L.append(f"| {rid} | {arch} | ? | pending |")
            continue
        task = j["config"]["task"]
        if eb and lb:
            re_, rl = ratios(j, eb, lb)
            c4 = f"{f(re_, 3)}/{f(rl, 3)} ({mark(c4_pass(re_, rl))})"
        else:
            c4 = "-"
        L.append(f"| {rid} | {arch} | {task} | {f(h['acc'])} | {f(h['c1_var_p'])} "
                 f"| {f(h['c2_r_measured'])} | {f(h['c3_kl'])} | {c4} |")
    L.append("")


def sec_analysis(L):
    L.append("## Mode analysis (main runs)")
    L.append("")
    for task, rid in MAIN_RUNS.items():
        j = load(rid)
        h = hard(j)
        if h is None:
            L.append(f"### {task}: pending\n")
            continue
        prof = h["stage_profile"]
        L.append(f"### {task} ({rid})")
        L.append("")
        so = j["test"]["soft"]
        L.append(f"acc={f(h['acc'])}, emits/input={f(h['mean_emits'],2)}, "
                 f"p_eff(soft)={f(so['p_eff'])}, p_eff(hard)={f(h['p_eff'])}, "
                 f"c3 KL={f(h['c3_kl'])}")
        L.append("")
        L.append("mean p_emit per stage:")
        L.append("")
        L.append("`" + " ".join(f"{p:.2f}" for p in prof) + "`")
        L.append("")
        L.append("emissions by measured difficulty:")
        L.append("")
        L.append("| measured | n | emits | mean p |")
        L.append("|---|---|---|---|")
        for m in sorted(h["emits_by_difficulty"]):
            b = h["emits_by_difficulty"][m]
            L.append(f"| {m} | {b['n']} | {f(b['mean_emits'], 2)} | {f(b['mean_p'])} |")
        L.append("")
        L.append(f"corr(measured, emits) = {f(h['c2_r_measured'])} ; "
                 f"corr(label, emits) = {f(h['c2_r_label'])} "
                 f"({'disagree' if abs(h['c2_r_measured']-h['c2_r_label'])>0.05 else 'agree'}).")
        L.append(f"Contribution of each mode (hybrid hard vs forced): "
                 f"latent gap = {f(j['test']['latent_gap'])}, "
                 f"emit gap = {f(j['test']['emit_gap'])}.")
        L.append("")


def sec_phase2(L):
    L.append("## Phase 2 - pressure regimes")
    L.append("")
    L.append("Regimes designed to FORCE externalizing state (hypotheses in "
             "`experiments/PHASE2.md`). r_emit / r_latent = hybrid test CE ÷ "
             "specialist test CE re-trained in the SAME regime (tolerance "
             f"{TOL}); schedule rows have no specialists and are compared "
             "against the controller in the summary lines below.")
    L.append("")
    L.append("| run | regime | acc | ce | c1 | c2 | c3 KL | emits | r_emit | r_latent |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    jmain = load("A_mlp_main")
    hm = hard(jmain)
    re_ref = rl_ref = None
    if hm is not None:
        re_ref, rl_ref = ratios(jmain, "A_emit_base", "A_latent_base")
    data = {}
    for rid, regime, eb, lb in PHASE2_REGIMES:
        j = load(rid)
        h = hard(j)
        if h is None:
            data[rid] = None
            L.append(f"| {rid} | {regime} | pending |")
            continue
        re_ = rl = None
        if eb and lb:
            re_, rl = ratios(j, eb, lb)
        data[rid] = {"h": h, "re": re_, "rl": rl}
        re_s = f(re_, 3) if (eb and lb) else "-"
        rl_s = f(rl, 3) if (eb and lb) else "-"
        L.append(f"| {rid} | {regime} | {f(h['acc'])} | {f(h['ce'])} | {f(h['c1_var_p'])} "
                 f"| {f(h['c2_r_measured'])} | {f(h['c3_kl'])} | {f(h['mean_emits'], 2)} "
                 f"| {re_s} | {rl_s} |")
    L.append("")

    def st(rid, key):
        d = data.get(rid)
        return d[key] if d else None

    # -- H1: lossy channel ---------------------------------------------------
    rl03, rl05 = st("A_s03", "rl"), st("A_s05", "rl")
    if rl_ref is None or rl03 is None or rl05 is None:
        v1 = "pending"
    else:
        ok = (rl03 < rl_ref and rl05 <= rl03
              and (c4_pass(st("A_s03", "re"), rl03) or c4_pass(st("A_s05", "re"), rl05)))
        v1 = "confirmed" if ok else "refuted"
    L.append(f"- **H1 (lossy channel)**: r_latent at sigma=0.3 -> {f(rl03, 3)}, at sigma=0.5 -> "
             f"{f(rl05, 3)}; phase-1 reference (sigma=0.1) r_emit/r_latent = "
             f"{f(re_ref, 3)}/{f(rl_ref, 3)}. Prediction: r_latent falls as sigma rises and "
             f"c4 <= {TOL} at some sigma. Verdict: **{v1}**.")
    # -- H2: information over time --------------------------------------------
    rl_b, c2_b = st("A_bneck", "rl"), st("A_bneck", "h")
    c2_b = c2_b["c2_r_measured"] if c2_b else None
    c2_ref = hm["c2_r_measured"] if hm is not None else None
    if rl_b is None or rl_ref is None or c2_b is None or c2_ref is None:
        v2 = "pending"
    else:
        v2 = "confirmed" if (rl_b < rl_ref and c2_b > c2_ref) else "refuted"
    L.append(f"- **H2 (information over time)**: r_latent with the gradual reader -> {f(rl_b, 3)} "
             f"vs phase-1 {f(rl_ref, 3)}; c2 measured {f(c2_b, 3)} vs phase-1 {f(c2_ref, 3)}. "
             f"Prediction: r_latent improves AND c2 strengthens. Verdict: **{v2}**.")
    # -- H3: controller vs fixed schedule --------------------------------------
    ce_main = hm["ce"] if hm is not None else None
    em_main = hm["mean_emits"] if hm is not None else None
    h4, h2 = st("A_sched4", "h"), st("A_sched2", "h")
    ce4 = h4["ce"] if h4 else None
    ce2 = h2["ce"] if h2 else None
    em4 = h4["mean_emits"] if h4 else None
    em2 = h2["mean_emits"] if h2 else None
    if ce_main is None or ce4 is None:
        v3 = "pending"
    else:
        v3 = "confirmed" if ce_main <= TOL * ce4 else "refuted"
    L.append(f"- **H3 (controller vs fixed schedule)**: controller CE {f(ce_main)} "
             f"({f(em_main, 2)} emits) vs fixed schedule k=4 CE {f(ce4)} ({f(em4, 2)} emits) "
             f"and k=2 CE {f(ce2)} ({f(em2, 2)} emits). Prediction: the controller beats the "
             f"budget-matched schedule (k=4, within {TOL}). Verdict: **{v3}**.")
    L.append("")
    L.append("H4 (deep tasks, depth 8-12, d=192) is held back until H1/H2 show signal; "
             "schedule-run emit counts are asserted to equal N/k (±0.5) by "
             "`tests/test_09_phase2.py`.")
    L.append("")


def sec_repro(L):
    L.append("## Reproducibility")
    L.append("")
    meta_p = os.path.join(common.RESULTS, "_meta.json")
    meta = common.read_json(meta_p) if os.path.exists(meta_p) else {}
    n_tests = meta.get("pytest_count", "n/a")
    L.append(f"- All numbers above are rendered by `scripts/render_findings.py` from "
             f"`results/*.json`. Regenerate with `python -m scripts.render_findings`.")
    L.append(f"- Test suite: `pytest -q` (collection count reported here: **{n_tests}** "
             f"; a test asserts this matches live `pytest --collect-only`).")
    L.append(f"- Data is generated deterministically from fixed seeds (data seed 1234); "
             f"each run's model seed is in its JSON config.")
    L.append(f"- Push rule: every run commits+pushes every 250 steps (spec "
             f"floor 500; `--push-every` in each run's JSON); "
             f"`scripts/recover_git.sh` refuses unpushed state (git history shows the cadence).")
    rid = list(MAIN_RUNS.values())[0]
    j = load(rid)
    if j:
        c = j["config"]
        flags = ["task", "arch", "d", "n", "vocab", "steps", "batch", "sigma",
                 "lam_price", "lam_commit", "lam_nd", "q_conf", "t_conf",
                 "tau", "semantics", "force", "depth_max", "p_trivial",
                 "reader_layers", "train_n", "dev_n", "test_n"]
        cli = " ".join(f"--{k.replace('_', '-')} {c[k]}" for k in flags if k in c and c[k] is not None)
        L.append(f"- Reproduce one run: `python -m scripts.train --run-id {rid} {cli}` "
                 f"(push rule on; add `--no-git` to skip).")
    else:
        L.append(f"- Reproduce one run: `python -m scripts.train --run-id {rid} ...` "
                 f"(see the run's JSON config once it exists).")
    L.append("")


def render():
    L = ["# FINDINGS", "",
         "Living document -- **regenerated from `results/*.json` only** by "
         "`scripts/render_findings.py`. Numbers below are not hand-editable.",
         ""]
    sec_tldr(L)
    sec_negative(L)
    sec_criteria(L)
    sec_scaling(L)
    sec_arch(L)
    sec_analysis(L)
    sec_phase2(L)
    sec_repro(L)
    L.append("---")
    L.append("*Negative results are first-class: a refuted expectation above is a finding, "
             "not a bug.*")
    return "\n".join(L) + "\n"


def render_readme_block():
    L = ["| task | acc (hard) | emits/input | c2 corr(measured) | status |",
         "|---|---|---|---|---|"]
    for task, rid in MAIN_RUNS.items():
        j = load(rid)
        h = hard(j)
        if h is None:
            L.append(f"| {task} | pending | | | running |")
            continue
        L.append(f"| {task} | {f(h['acc'])} | {f(h['mean_emits'], 2)} "
                 f"| {f(h['c2_r_measured'])} | done |")
    return "\n".join(L)


def main():
    common.setup_paths()
    text = render()
    out = os.path.join(common.REPO_ROOT, "FINDINGS.md")
    with open(out, "w") as fh:
        fh.write(text)

    # README results block (between markers)
    rpath = os.path.join(common.REPO_ROOT, "README.md")
    block = render_readme_block()
    with open(rpath) as fh:
        readme = fh.read()
    begin, end = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    if begin in readme and end in readme:
        pre, rest = readme.split(begin, 1)
        post = rest.split(end, 1)[1]
        with open(rpath, "w") as fh:
            fh.write(pre + begin + "\n" + block + "\n" + end + post)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
