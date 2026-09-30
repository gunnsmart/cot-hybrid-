# Deliverable

The full deliverable for the "interleaved latent-token reasoning" research
task is in `interleaved-reasoning/` — tracked in this repository on this
session branch. Every training run commits+pushes it here every 500 steps
(see `interleaved-reasoning/PROBLEM.md` §7).

To publish it to your own account, make a standalone repo from the
directory (see `interleaved-reasoning/README.md`, "Publishing"):

    cd interleaved-reasoning
    git init -b main && git add -A && git commit -m "interleaved reasoning"
    git remote add origin https://github.com/<you>/interleaved-reasoning.git
    git push -u origin main
