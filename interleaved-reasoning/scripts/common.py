"""Shared helpers: repo paths, JSON results I/O, and the git push rule.

This deliverable is tracked inside the session repository (the outer git
repository that contains this directory). All git commands run with cwd set
to the deliverable root; git resolves the surrounding repository.

Push rule (mandatory, see PROBLEM.md §7):
  * commit + push after every 250 training steps (spec floor: 500)
  * never run > 500 steps without pushing
  * training REFUSES to continue if a push fails
  * scripts/recover_git.sh refuses if unpushed commits exist
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")
RUNS = os.path.join(REPO_ROOT, "runs")


def setup_paths():
    for p in (RESULTS, CHECKPOINTS, RUNS):
        os.makedirs(p, exist_ok=True)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def run_json_path(run_id):
    return os.path.join(RESULTS, f"{run_id}.json")


def latest_pt_path(run_id):
    d = os.path.join(RUNS, run_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "latest.pt")


def best_pt_path(run_id):
    return os.path.join(CHECKPOINTS, f"{run_id}.pt")


# ---------------------------------------------------------------------------
# Push rule
# ---------------------------------------------------------------------------

def git(cwd=REPO_ROOT):
    def _(*args, check=True):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r
    return _


def push_step(run_id: str, step: int, push_every: int, dry_run: bool = False):
    """Commit+push the run's state every `push_every` steps.

    Commits the run's JSON (progress + metrics) and, when it exists, the
    current best checkpoint; pushes the current branch. The full fp16
    snapshot of the latest weights is written to runs/<id>/latest.pt on disk
    at the same moment (scratch dir, used for resume).

    Raises on any failure; the caller must refuse to continue training.
    """
    if dry_run:
        return True
    if step % push_every != 0:
        return True
    g = git()
    files = [f"results/{run_id}.json"]
    if os.path.exists(best_pt_path(run_id)):
        files.append(f"checkpoints/{run_id}.pt")
    g("add", "--", *files)
    g("commit", "-m", f"{run_id}: step {step} (push rule: every {push_every} steps)",
      check=False)  # may find nothing staged if state did not change
    g("push", "origin", "HEAD")  # raises on failure -> train() exits
    # Keep the local remote-tracking ref in sync (best effort: some
    # environments do not update it on push; the push itself already
    # succeeded, so a failed re-fetch must not abort training).
    r = g("rev-parse", "--abbrev-ref", "HEAD")
    g("fetch", "origin", r.stdout.strip(), check=False)
    return True
