#!/usr/bin/env bash
# recover_git.sh -- pre-flight guard for the push rule.
#
#   * refuses (exit 1) if this repository has ANY unpushed commits,
#   * refuses if the upstream (origin/<current branch>) does not exist yet,
#   * warns about uncommitted changes to results/ or checkpoints/.
#
# Run it before resuming training or before starting a new run:
#     scripts/recover_git.sh
set -eu
# pipefail where the shell supports it (bash/zsh/ksh93); plain sh (dash) lacks it
if (set -o pipefail) 2>/dev/null; then set -o pipefail; fi
cd "$(dirname "$0")/.."

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if ! git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  echo "REFUSING: upstream origin/$BRANCH does not exist yet; push first." >&2
  exit 1
fi
N="$(git rev-list --count "origin/$BRANCH..HEAD")"
if [ "$N" -gt 0 ]; then
  echo "REFUSING: $N unpushed commit(s) on $BRANCH." >&2
  echo "Push rule: never run >500 steps without pushing. Fix with:" >&2
  echo "  git push origin $BRANCH" >&2
  git --no-pager log --oneline "origin/$BRANCH..HEAD" >&2
  exit 1
fi
if ! git diff --quiet -- results checkpoints || ! git diff --cached --quiet -- results checkpoints; then
  echo "WARNING: uncommitted changes under results/ or checkpoints/ (commit+push before training)." >&2
fi
echo "OK: $BRANCH is clean and fully pushed ($(git rev-parse --short HEAD))."
