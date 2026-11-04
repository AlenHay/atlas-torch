"""Program registry of the compat API (fast tier): the torch programs are registered under the BasicATLAS
executable names, and the SYNBEG adapter echoes its card exactly as synbeg.exe (validate_run() parses it)."""
import os
from pathlib import Path

import pytest

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def test_torch_programs_registered():
    from atlas_torch.compat import runners
    reg = runners.registered()
    expected = {"atlas9mem", "kappa9", "xnfpelsyn", "synbeg", "rgfalllinesnew", "rmolecasc", "rh2ofast", "synthe",
                "spectrv", "xnfdf", "dfsynthe", "dfsortp", "separatedf", "kapreadts"}
    assert expected <= set(reg), expected - set(reg)
    from atlas_torch.atlas9.program import atlas9mem_program
    assert reg["atlas9mem"] is atlas9mem_program


def test_atlas9mem_backend_torch_has_no_fallback(monkeypatch, tmp_path):
    """ATLAS_TORCH_BACKEND=torch dispatches to the torch runner (never the executable)."""
    from atlas_torch.compat import runners
    seen = {}
    monkeypatch.setenv("ATLAS_TORCH_BACKEND", "torch")
    runners._discover()
    monkeypatch.setitem(runners._registry, "atlas9mem", lambda wd, text, units: seen.setdefault("call", (wd, text)) and "ok")
    assert runners.run_program("atlas9mem", str(tmp_path), "END\n", {}) == "ok"
    assert seen["call"] == (str(tmp_path), "END\n")


@pytest.mark.parametrize("window", [0, 1, 2, 3])
def test_synbeg_stdout_equals_stock(window, tmp_path):
    from atlas_torch.compat.torch_programs import synbeg_program
    ref = EXT / "runs/stock/atlas/sun" / f"window_{window}" / "synbeg.out"
    if not ref.exists():
        pytest.skip("stock SYNTHE reference run not found")
    stock = ref.read_text()
    # the card BasicATLAS writes (compat.atlas._synthe_script) for the window's WLBEG/WLEND
    w = stock.split()
    lo, hi = float(w[w.index("WLBEG=") + 1]), float(w[w.index("WLEND=") + 1])
    card = ("{:<3s}       {:<10.4f}{:<10.4f}{:<9.2f} {:<10.4f}{:<3d}{:<7d}{:<10.5f}{:<5d}{:<5d}"
            .format("VAC", lo, hi, 600000.0, 1.5, 0, -1, 0.0001, 1, 0))
    out = synbeg_program(str(tmp_path), card + "\nAIRorVAC  WLBEG     WLEND     RESOLU    TURBV  IFNLTE LINOUT\n", {})
    assert out == stock
    assert all((tmp_path / f"fort.{u}").exists() for u in (12, 14, 19, 20, 93))
