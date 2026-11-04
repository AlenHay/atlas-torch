"""Shared fixtures: external (never committed) data, Fortran oracles and golden traces."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
ORACLE_DIR = EXT / "oracle"
TRACE_DIR = EXT / "traces"


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"missing external artifact {p} (run M0 scripts)")
    return p


@pytest.fixture(scope="session")
def ext():
    return _need(EXT)


@pytest.fixture(scope="session")
def oracle():
    """oracle('atlas7v') -> Oracle over $ATLAS_EXT/oracle/libatlas7v.so with load-time SAVE state.

    By default one private copy per library is shared by the whole session and rewound to its load-time
    image (Oracle.restore: all COMMONs and SAVEd/DATA locals) on every call, which is equivalent to a fresh
    load for the library's own state; a previously returned handle to the same library is rewound too.
    oracle(name, fresh=True) loads an independent copy (needed only to hold two instances of one library
    at once or to get a pristine mapping; libgfortran/libifcore I/O-unit state is process-wide either way).
    """
    from atlas_torch.testing.fortran_oracle import Oracle

    shared = {}

    def make(name, fresh=False):
        path = _need(ORACLE_DIR / f"lib{name}.so")
        if name in ("xnfdf", "dfsynthe", "kappa9"):       # ifx builds: Intel runtime must be preloaded
            from atlas_torch.dfsynthe.xnf.oracle_support import ensure_ifx_runtime
            ensure_ifx_runtime()
        if fresh:
            return Oracle(path)
        if name not in shared:
            shared[name] = Oracle(path, snapshot=True)
        return shared[name].restore()
    return make


@pytest.fixture(scope="session")
def _xnfdf_solar_deck():
    from atlas_torch.dfsynthe.xnf.readin import Readin
    run = _need(TRACE_DIR / "odf_solar" / "xw")
    r = Readin((run / "stdin.txt").read_text(), (run / "fort.2").read_text())
    return [r.read(1) for _ in range(57)]


@pytest.fixture(scope="module")
def xnfdf_solar_models(_xnfdf_solar_deck):
    """The 57 models of the traced DFSYNTHE solar grid (xnfdf READIN, ~3 s), read once per session; each
    test module gets its own deep copy."""
    import copy
    return copy.deepcopy(_xnfdf_solar_deck)


@pytest.fixture(scope="session")
def golden():
    """golden('sun_500_510/xnfpelsyn', 'pops', hit, 'entry'|'exit') -> dict of numpy arrays from the gdb trace."""
    def load(stage, routine, hit=0, phase="entry"):
        p = _need(TRACE_DIR / stage / f"{routine.lower()}_{hit}_{phase}.npz")
        with np.load(p) as d:
            return {k: d[k] for k in d.files}
    return load


_STAGE_DIR = Path(__file__).parent / "stage"
FAST_EXAMPLES = int(os.environ.get("ATLAS_FAST_EXAMPLES", "10"))


def _fast_tier(config) -> bool:
    """The pre-commit fast tier is `pytest -m "not stage and not full" tests`."""
    return "not stage" in (config.getoption("markexpr") or "")


def pytest_report_header(config):
    if _fast_tier(config) and FAST_EXAMPLES > 0:
        return (f"fast tier: hypothesis max_examples capped at {FAST_EXAMPLES} (explicit @example cases always run); "
                "declared counts run in any session that does not deselect stage (ATLAS_FAST_EXAMPLES=0 disables the cap)")


def pytest_collection_modifyitems(config, items):
    """Every test under tests/stage/ belongs to the stage tier, whether or not its module says so.

    In the fast tier, property tests run with at most FAST_EXAMPLES hypothesis examples: the oracle
    comparison of every routine still runs on random inputs, while the declared (thorough) counts run in
    full sessions (`pytest tests`, or any run of a single file).
    """
    for item in items:
        if _STAGE_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.stage)
    if not _fast_tier(config) or FAST_EXAMPLES <= 0:
        return
    from hypothesis import settings
    for item in items:
        fn = getattr(item, "function", None)
        cur = getattr(fn, "_hypothesis_internal_use_settings", None)
        if getattr(fn, "is_hypothesis_test", False) and cur is not None and cur.max_examples > FAST_EXAMPLES:
            fn._hypothesis_internal_use_settings = settings(cur, max_examples=FAST_EXAMPLES)
