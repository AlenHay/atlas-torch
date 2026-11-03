PY := .venv/bin/python
DX := scripts/dx

.PHONY: fast stage full score noise reference oracles

fast:            ## fast tier (< ~2 min): unit + oracle tests
	$(DX) '$(PY) -m pytest -q -m "not stage and not full" -x --timeout=300 tests'

stage:           ## stage tier: per-milestone end-to-end checks
	$(DX) '$(PY) -m pytest -q -m stage tests'

full:            ## full tier: everything
	$(DX) '$(PY) -m pytest -q tests'

score:           ## scoreboard vs frozen tolerances -> results/scoreboard.jsonl, RESULTS.md
	$(DX) '$(PY) tools/score/score.py $(ARGS)'

noise:           ## (M0) measure Fortran noise floor; `make noise ARGS=--write` freezes results/tolerances.json once
	$(DX) '$(PY) tools/score/noise_floor.py $(ARGS)'

reference:       ## (M0) all Fortran reference + noise-variant runs
	$(DX) scripts/reference_queue.sh

oracles:         ## build -g shared libraries for ctypes oracles
	$(DX) tools/oracle/build_libs.sh
