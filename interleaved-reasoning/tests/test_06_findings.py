"""FINDINGS.md is generated from JSON only: re-rendering must reproduce it
byte-for-byte, the README results block must match, and the reported pytest
count must equal the live collection count."""
import os
import re
import subprocess
import sys

import pytest

from scripts import common
from scripts.render_findings import render, render_readme_block


def test_findings_renders_from_json_stably():
    common.setup_paths()
    text = render()
    with open(os.path.join(common.REPO_ROOT, "FINDINGS.md")) as f:
        committed = f.read()
    assert committed == text, (
        "FINDINGS.md is stale: run `python -m scripts.render_findings` "
        "(every number must come from results/*.json)")


def test_readme_block_matches():
    block = render_readme_block()
    with open(os.path.join(common.REPO_ROOT, "README.md")) as f:
        readme = f.read()
    begin, end = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    assert begin in readme and end in readme
    between = readme.split(begin, 1)[1].split(end, 1)[0]
    assert between.strip() == block.strip(), "README results block drifted from JSON"


def _live_count():
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=common.REPO_ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    m = re.search(r"(\d+) tests? collected", r.stdout)
    assert m, f"could not parse collection count:\n{r.stdout[-500:]}"
    return int(m.group(1))


def test_pytest_count_matches_findings():
    meta_p = os.path.join(common.RESULTS, "_meta.json")
    if not os.path.exists(meta_p):
        pytest.skip("_meta.json not written yet (finalize step)")
    meta = common.read_json(meta_p)
    n = meta["pytest_count"]
    with open(os.path.join(common.REPO_ROOT, "FINDINGS.md")) as f:
        findings = f.read()
    assert f"**{n}**" in findings, "FINDINGS does not report the meta count"
    live = _live_count()
    assert n == live, f"FINDINGS reports {n} tests but live collection is {live}"
