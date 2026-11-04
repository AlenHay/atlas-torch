"""The ctypes oracle toolkit itself: private copies, restore() to the load-time image, cleanup, DEEPBIND."""
import ctypes
import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atlas_torch.testing.fortran_oracle import Oracle, oracle_tmp_root

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
LIB = EXT / "oracle" / "libatlas7v.so"
pytestmark = pytest.mark.skipif(not LIB.exists(), reason="oracle library missing")


def test_restore_rewinds_commons_and_saved():
    with pytest.raises(RuntimeError):
        Oracle(LIB).restore()
    o = Oracle(LIB, snapshot=True)
    blk = next(iter(o.commons_layout))
    var, view = next((k, v) for k, v in o.common(blk).items() if v.dtype.kind == "f")
    before = view.copy()
    view[...] = 12345.0
    o2 = Oracle(LIB)                                  # independent copy: not affected
    assert np.array_equal(o2.common(blk)[var], before)
    o.restore()
    assert np.array_equal(o.common(blk)[var], before)


def test_copies_live_in_scratch_and_are_removed():
    o = Oracle(LIB)
    d = Path(o._tmp)
    assert d.parent == oracle_tmp_root() and d.name.startswith(f"oracle_{os.getpid()}_") and d.exists()
    o.reset()
    assert not d.exists() and Path(o._tmp).exists()
    d = Path(o._tmp)
    del o
    gc.collect()
    assert not d.exists()


def test_no_leftovers_or_errors_at_interpreter_exit():
    code = ("import sys; from atlas_torch.testing.fortran_oracle import Oracle; "
            f"o = Oracle({str(LIB)!r}); print(o._tmp); sys.stdout.flush()")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stderr == "", r.stderr
    assert not Path(r.stdout.strip()).exists()


def test_ifx_libraries_bind_libimf():
    """deepbind=True resolves LOG10F like the stock ifx executables (libimf before libm in DT_NEEDED)."""
    from atlas_torch.dfsynthe.xnf import oracle_support as osup
    if not osup.available():
        pytest.skip("libxnfdf.so or the Intel runtime missing")
    osup.ensure_ifx_runtime()
    o = Oracle(osup.oracle_path(), deepbind=True)
    # PFIRON(20,1,4.0,10**3.760881766617063): glibc 2.39 log10f(5766.0947) is 1 ulp above libimf's
    args = (20, 1, 4.0, 10 ** 3.760881766617063, ctypes.c_double(0.))
    assert o.call("pfiron", *args)[1][-1] == 3.7922563552856445
    args[-1].value = 0.
    assert Oracle(osup.oracle_path()).call("pfiron", *args)[1][-1] == 3.7922558784484863   # glibc log10f
