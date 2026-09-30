#!/usr/bin/env bash
# finalize.sh -- end-of-project pipeline:
#   tests -> meta (pytest count) -> render FINDINGS from JSON -> tests again
#   -> idempotency check -> commit + push.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

scripts/recover_git.sh || { echo "ABORT: unpushed state."; exit 1; }

echo "== pytest (pre-render) =="
"$PY" -m pytest -q 2>&1 | tail -3 || true

COUNT=$("$PY" -m pytest --collect-only -q 2>/dev/null \
        | grep -oE '[0-9]+ tests? collected' | grep -oE '^[0-9]+' | head -1)
[ -n "${COUNT:-}" ] || { echo "ABORT: could not get pytest count"; exit 1; }

"$PY" - "$COUNT" <<'EOF'
import sys
from scripts import common
common.setup_paths()
common.write_json(common.run_json_path("_meta"),
                  {"pytest_count": int(sys.argv[1]),
                   "generated_by": "scripts/finalize.sh",
                   "generated_at": common.now_iso()})
EOF
echo "meta: pytest_count=$COUNT"

echo "== render FINDINGS from JSON =="
"$PY" -m scripts.render_findings

echo "== pytest (post-render, must be green) =="
"$PY" -m pytest -q || { echo "ABORT: tests failing."; exit 1; }

echo "== idempotency =="
cp FINDINGS.md /tmp/findings.a && cp README.md /tmp/readme.a
"$PY" -m scripts.render_findings
diff -q /tmp/findings.a FINDINGS.md && diff -q /tmp/readme.a README.md \
    || { echo "ABORT: render not idempotent."; exit 1; }

echo "== commit + push =="
git add -A
git commit -m "final: results JSON, FINDINGS (rendered), tests" --allow-empty
git push origin main
echo "FINALIZE OK"
