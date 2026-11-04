"""Fast-tier unit checks of PROGRAM SPECTRV pieces (moved from tests/stage/test_spectrv_prog.py, which holds
the whole-program runs): SUBROUTINE W output vs the Fortran, the Fw.d reader, REAL*8**INTEGER (__powidf2) and
the WLEDGE edge index."""
import ctypes

import numpy as np
import torch

from atlas_torch.synthe import spectrv_prog as sp
from atlas_torch.synthe.josh import w as w_format


def test_w_vs_fortran(oracle, capfd):
    """SUBROUTINE W(A,B,N) output (1P10E12.4, 3-digit exponents, negatives) vs the Fortran's unit 6."""
    o = oracle("spectrv")
    gf = ctypes.CDLL("libgfortran.so.5")
    # Unit 6 is libgfortran's process-wide buffer: Fortran routines called through other oracles earlier in
    # the session (e.g. an "NMERGE EMERGE EMERGEH" table) may still sit in it unflushed and would otherwise
    # surface in this test's capture. Flush and discard it first.
    gf._gfortran_flush_i4(ctypes.byref(ctypes.c_int(6)))
    capfd.readouterr()
    rng = np.random.default_rng(5)
    for n in (0, 1, 10, 11, 25, 72):
        b = np.asfortranarray(rng.normal(size=max(n, 1)) * 10.0 ** rng.integers(-150, 150, max(n, 1)))
        if n > 3:
            b[1] = 0.0; b[2] = -9.99995e-5; b[3] = 9.99995e104
        fn = o.lib.w_
        fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
        fn(b"LABEL1", b.ctypes.data, ctypes.byref(ctypes.c_int(n)), 6)
        gf._gfortran_flush_i4(ctypes.byref(ctypes.c_int(6)))
        got = capfd.readouterr().out
        assert got == w_format("LABEL1", b, n), n


def test_read_fw_d():
    assert sp.read_fw_d("0.0       ", 5) == 0.0
    assert sp.read_fw_d("        1.", 5) == 1.0
    assert sp.read_fw_d("    12345 ", 5) == 0.12345
    assert sp.read_fw_d("  1.5E-02 ", 5) == 0.015
    assert sp.read_fw_d("          ", 5) == 0.0


def test_powi_is_libgcc_powidf2():
    """WAVE = WBEGIN*RATIO**(NU-1): gfortran REAL*8**INTEGER calls libgcc __powidf2."""
    lib = ctypes.CDLL("libgcc_s.so.1")
    f = lib.__powidf2
    f.restype = ctypes.c_double; f.argtypes = [ctypes.c_double, ctypes.c_int]
    rng = np.random.default_rng(1)
    for res in (600000.0, 300000.0, 100000.0, 1234567.0):
        ratio = 1.0 + 1.0 / res
        m = np.r_[np.arange(0, 70), rng.integers(0, 2_010_001, 3000)]
        got = sp.powi(ratio, torch.as_tensor(m)).numpy()
        exp = np.array([f(ratio, int(k)) for k in m])
        assert np.array_equal(got, exp)
        assert all(sp.powi_scalar(ratio, int(k)) == f(ratio, int(k)) for k in m[:200])


def test_edge_index_matches_sequential_scan():
    rng = np.random.default_rng(0)
    wl = np.r_[np.sort(rng.uniform(0, 100, 40)), np.zeros(337)]
    w = np.sort(rng.uniform(wl[0], wl[39] - 1e-9, 5000))
    w = np.sort(np.r_[w, wl[7], wl[7], wl[20]])          # points exactly on an edge

    def seq(wl):
        i, o = 0, []
        for x in w:
            while not (x < wl[i + 1]):
                i += 1
            o.append(i)
        return np.array(o)
    assert np.array_equal(sp.edge_index(torch.as_tensor(w), wl, 40).numpy(), seq(wl))
    wl2 = wl.copy(); wl2[5], wl2[6] = wl2[6], wl2[5]     # non-monotonic WLEDGE: sequential fallback
    assert np.array_equal(sp.edge_index(torch.as_tensor(w), wl2, 40).numpy(), seq(wl2))
