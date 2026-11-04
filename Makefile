PY := .venv/bin/python
# inside the atlas-dev container run directly; on the host go through the container
DX := $(if $(CONTAINER_ID),bash -c,scripts/dx)

.PHONY: fast stage full score noise reference oracles native

fast:            ## fast tier (< ~2 min): unit + oracle tests
	$(DX) '$(PY) -m pytest -q -m "not stage and not full" -x --timeout=300 tests'

stage:           ## stage tier: per-milestone end-to-end checks
	$(DX) '$(PY) -m pytest -q -m stage tests'

full:            ## full tier: everything
	$(DX) '$(PY) -m pytest -q tests'

native:          ## build the DFSYNTHE line-accumulation C kernel (bit-identical to the torch fallback)
	$(DX) '$(PY) -m atlas_torch.dfsynthe.lines.native'

score: native    ## scoreboard vs frozen tolerances -> results/scoreboard.jsonl, RESULTS.md
	$(DX) '$(PY) tools/score/score.py $(ARGS)'

noise:           ## (M0) measure Fortran noise floor; `make noise ARGS=--write` freezes results/tolerances.json once
	$(DX) '$(PY) tools/score/noise_floor.py $(ARGS)'

reference:       ## (M0) all Fortran reference + noise-variant runs
	$(DX) scripts/reference_queue.sh

oracles:         ## build -g shared libraries for ctypes oracles
	$(DX) tools/oracle/build_libs.sh

fmap:            ## Fortran map + variant table -> results/fmap/ (not committed: quotes Kurucz source)
	mkdir -p results/fmap
	for p in xnfdf dfsynthe kappa9 atlas9mem atlas7v xnfpelsyn synthe spectrv rgfalllinesnew rmolecasc rpredict rh2ofast synbeg; do \
	  $(PY) tools/fmap.py map $$HOME/atlas-ext/BasicATLAS/src/$$p.for --json results/fmap/$$p.json; done
	$(PY) tools/fmap.py variants $$HOME/atlas-ext/BasicATLAS/src/{xnfdf,kappa9,atlas9mem,atlas7v}.for --json results/fmap/variants.json --md results/fmap/variants.md
