PY ?= python3
VENV := .venv
VPY := $(VENV)/bin/python

.PHONY: setup run-matrix test findings recover final

setup:
	python3 -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install torch numpy pytest

run-matrix:
	scripts/run_matrix.sh

test:
	$(VPY) -m pytest -q

findings:
	$(VPY) -m scripts.render_findings

recover:
	scripts/recover_git.sh

final:
	scripts/finalize.sh
