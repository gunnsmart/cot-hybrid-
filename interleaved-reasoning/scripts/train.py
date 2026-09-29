"""End-to-end single-phase training of the interleaved processor.

Usage (from the repo root):
    python -m scripts.train --task arithmetic --arch mlp --run-id A_mlp_main \
        [--d 96] [--n 12] [--steps 3000] [--seed 0] ...

Mandatory push rule enforced here:
    * every `--push-every` (default 500) steps: commit + push to origin/main
    * if the push fails, training REFUSES to continue (exit code 2)
    * `scripts/recover_git.sh` refuses to start any work with unpushed commits
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from scripts import common
from src.data import make_dataset, collate
from src.models import InterleavedProcessor
from src.metrics import evaluate, full_test_eval, save_checkpoint, load_checkpoint

DEVICE = "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["arithmetic", "logic", "recall"])
    p.add_argument("--arch", default="mlp", choices=["mlp", "gru", "ssm", "transformer"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.1,
                   help="latent channel noise (fixed channel property)")
    p.add_argument("--lam-price", type=float, default=0.2, help="emission price weight")
    p.add_argument("--lam-commit", type=float, default=0.1, help="mode commitment weight")
    p.add_argument("--lam-nd", type=float, default=5.0, help="non-degeneracy weight")
    p.add_argument("--v-min", type=float, default=0.1, help="variance floor (criterion 1 threshold)")
    p.add_argument("--tau", type=float, default=0.5, help="hard decision threshold at inference")
    p.add_argument("--semantics", default="additive", choices=["additive", "reset"])
    p.add_argument("--force", default="none", choices=["none", "emit", "latent"],
                   help="baseline: force one mode (no mode mechanism trained)")
    p.add_argument("--reader-layers", type=int, default=1, choices=[1, 2],
                   help="input reader depth (task interface, not a stage)")
    p.add_argument("--train-n", type=int, default=4096)
    p.add_argument("--dev-n", type=int, default=512)
    p.add_argument("--test-n", type=int, default=1024)
    p.add_argument("--depth-max", type=int, default=None,
                   help="max task depth (calibration); default per task")
    p.add_argument("--p-trivial", type=float, default=0.30,
                   help="fraction of trivial surface components injected")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--push-every", type=int, default=250,
                   help="spec floor is 500 ('never run >500 steps without "
                        "pushing'); default 250 for finer live visibility")
    p.add_argument("--no-git", action="store_true", help="disable commit/push hook (tests)")
    p.add_argument("--resume", action="store_true", help="resume from runs/<run-id>/latest.pt")
    p.add_argument("--eval-batch", type=int, default=64)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_json(args):
    return {
        "run_id": args.run_id,
        "created": common.now_iso(),
        "status": "running",
        "config": {k: getattr(args, k) for k in
                   ["task", "arch", "d", "n", "vocab", "steps", "batch", "lr", "wd",
                    "seed", "sigma", "lam_price", "lam_commit", "lam_nd", "v_min",
                    "tau", "semantics", "force", "train_n", "dev_n", "test_n",
                    "depth_max", "p_trivial", "reader_layers",
                    "eval_every", "save_every", "push_every"]},
        "params": None,
        "curves": {"step": [], "train_loss": [], "dev_ce": [], "dev_acc": [],
                   "mean_p": []},
        "best_step": None,
        "test": None,
        "negative_result": None,
    }


def main():
    args = parse_args()
    common.setup_paths()
    set_seed(args.seed)
    torch.set_num_threads(2)

    force = None if args.force == "none" else args.force
    mechanism = force is None

    model = InterleavedProcessor(d=args.d, n_stages=args.n, vocab=args.vocab,
                                 arch=args.arch, mechanism=mechanism,
                                 sigma=args.sigma,
                                 reader_layers=args.reader_layers).to(DEVICE)
    common.write_json(common.run_json_path(args.run_id),
                      {**build_json(args),
                       "params": model.param_groups(),
                       "curves": build_json(args)["curves"]})

    train_ds = make_dataset(args.task, "train", args.train_n,
                            depth_max=args.depth_max, p_trivial=args.p_trivial)
    dev_ds = make_dataset(args.task, "dev", args.dev_n,
                          depth_max=args.depth_max, p_trivial=args.p_trivial)
    test_ds = make_dataset(args.task, "test", args.test_n,
                           depth_max=args.depth_max, p_trivial=args.p_trivial)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    step = 0
    best = (float("inf"), -1.0, 0)  # (ce, -acc, step)
    state = None
    if args.resume:
        sd = torch.load(common.latest_pt_path(args.run_id), map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        j = common.read_json(common.run_json_path(args.run_id))
        step = j["curves"]["step"][-1] if j["curves"]["step"] else 0

    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(train_ds))
    i = 0

    t0 = time.time()
    for step in range(step + 1, args.steps + 1):
        if i + args.batch > len(train_ds):
            order = rng.permutation(len(train_ds))
            i = 0
        idx = order[i:i + args.batch].tolist()
        i += args.batch
        X, y = collate([train_ds.samples[j] for j in idx], train_ds.max_len)

        logits, P, _ = model.predict(X, mode="soft", force=force,
                                     semantics=args.semantics)
        loss_task = F.cross_entropy(logits, y)

        if mechanism:
            loss_price = args.lam_price * P.mean()
            loss_commit = args.lam_commit * (4.0 * P * (1.0 - P)).mean()
            p_in = P.mean(dim=1)
            var_in = p_in.var()
            loss_nd = args.lam_nd * F.relu(args.v_min - var_in) ** 2
            loss = loss_task + loss_price + loss_commit + loss_nd
        else:
            loss = loss_task

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(model, dev_ds, list(range(len(dev_ds))), mode="hard",
                           tau=args.tau, force=force, semantics=args.semantics,
                           device=DEVICE, batch=args.eval_batch)
            j = common.read_json(common.run_json_path(args.run_id))
            c = j["curves"]
            c["step"].append(step)
            c["train_loss"].append(round(float(loss.item()), 5))
            c["dev_ce"].append(round(dev["ce"], 5))
            c["dev_acc"].append(round(dev["acc"], 5))
            c["mean_p"].append(round(float(P.mean()), 5))
            if (dev["ce"], -dev["acc"]) < (best[0], best[1]):
                best = (dev["ce"], -dev["acc"], step)
                j["best_step"] = step
                save_checkpoint(model, common.best_pt_path(args.run_id),
                                dtype=torch.float32)
            common.write_json(common.run_json_path(args.run_id), j)

        # ---- mandatory push rule -------------------------------------------
        if not args.no_git and (step % args.push_every == 0 or step == args.steps):
            save_checkpoint(model, common.latest_pt_path(args.run_id),
                            dtype=torch.float16)
            try:
                common.push_step(args.run_id, step, args.push_every)
            except RuntimeError as e:
                print(f"PUSH RULE VIOLATION: refusing to continue: {e}", file=sys.stderr)
                sys.exit(2)

        if step % 200 == 0:
            dt = time.time() - t0
            print(f"[{args.run_id}] step {step}/{args.steps} loss={loss.item():.4f} "
                  f"p={float(P.mean()):.3f} eta={dt/step*(args.steps-step)/60:.1f}m",
                  flush=True)

    # ---- final evaluation on the test set with the best checkpoint ---------
    best_model = InterleavedProcessor(d=args.d, n_stages=args.n, vocab=args.vocab,
                                      arch=args.arch, mechanism=mechanism,
                                      sigma=args.sigma,
                                      reader_layers=args.reader_layers).to(DEVICE)
    load_checkpoint(best_model, common.best_pt_path(args.run_id))
    test_res = full_test_eval(best_model, test_ds, tau=args.tau,
                              semantics=args.semantics, device=DEVICE,
                              batch=args.eval_batch, force=force)
    j = common.read_json(common.run_json_path(args.run_id))
    j["status"] = "done"
    j["finished"] = common.now_iso()
    j["seconds"] = round(time.time() - t0, 1)
    j["test"] = {k: v for k, v in test_res.items()
                 if k not in ("hard", "soft")} | {
        "hard": {k: v for k, v in test_res["hard"].items() if k != "counts"},
        "soft": {k: v for k, v in test_res["soft"].items() if k != "counts"},
    }
    common.write_json(common.run_json_path(args.run_id), j)
    if not args.no_git:
        save_checkpoint(best_model, common.latest_pt_path(args.run_id),
                        dtype=torch.float16)
        common.push_step(args.run_id, args.steps, args.push_every)

    h = j["test"]["hard"]
    print(f"DONE {args.run_id}: acc={h['acc']:.3f} ce={h['ce']:.3f} "
          f"emits={h['mean_emits']:.2f} c1={h['c1_var_p']:.3f} "
          f"c2m={h['c2_r_measured']:.3f} c3={h['c3_kl']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
