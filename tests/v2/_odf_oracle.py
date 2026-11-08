"""Validation inputs from the stock DFSYNTHE chain and the exact port (atlas_torch.dfsynthe): an OdfGas built from
the stock xnfpdf.dat (XNFDF's populations, Doppler widths and continuum) and the exact port's 0 km/s opacity
BUFFER of one temperature (SELECTLINES, XLINOP, TAPE12 lines, BUFFER = MAX(BUFFER, CONTINUUM))."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def odf_gas_from_xnfpdf(xnfpdf, iteff: int):
    """OdfGas of temperature iteff (1-based) from xnfpdf.dat: XNFPEL (n/U per NELION slot), DOPPLE, ABLOG (log10
    of the continuous absorption + scattering at the 3 frequencies per continua interval), T, P, XNE, RHO and the
    perturber density (XNFH + 0.42 XNFHE + 0.85 XNFH2)(T/1e4)^0.3."""
    from atlas_torch.dfsynthe.odf_io import read_xnfpdf
    from atlas_torch.v2.odf.gas import OdfGas
    from atlas_torch.v2.odf.lines import nelion_species
    from atlas_torch.v2.opacity.lines.opacity import GasState
    x = read_xnfpdf(xnfpdf, iteff)
    st = {k: torch.as_tensor(np.asarray(v, dtype=np.float64)) for k, v in x.state.items()}
    n = x.nrhox
    T, P, ne, rho = st["t"][:n], st["p"][:n], st["xne"][:n], st["rho"][:n]
    nou = torch.as_tensor(x.xnfpel.astype(np.float64))
    dop = torch.as_tensor(x.dopple.astype(np.float64))
    xnfhe = st["xnfhe"][:n]
    npert = (st["xnfh"][:n] + 0.42 * xnfhe[:, 0] + 0.85 * st["xnfh2"][:n]) * (T / 1e4) ** 0.3
    code, _, mol = nelion_species()
    atom = torch.zeros(n, 99, 10, dtype=torch.float64)
    for i in np.flatnonzero((code > 0) & ~mol):
        z = int(code[i])
        atom[:, z - 1, int(round((code[i] - z) * 100))] = nou[:, i]
    mcode = code[mol]
    state = GasState(T=T, n_e=ne, rho=rho, n_h1=st["xnfh"][:n], n_he1=xnfhe[:, 0], n_h2=st["xnfh2"][:n],
                     vturb=torch.zeros_like(T), n_over_u_atom=atom, n_over_u_mol=nou[:, mol], mol_codes=mcode,
                     batch_shape=(n,))
    ablog = torch.as_tensor(x.ablog.astype(np.float64)).reshape(n, -1)[:, :3 * (x.nedge - 1)]
    return OdfGas(T, P, rho, ne, npert, nou, dop, np.abs(x.wledge), ablog, state)


def exact_port_buffer(iteff: int, lo: int, hi: int, mix: str = "solar") -> dict:
    """The exact port's line opacity BUFFER (25, hi - lo + 1) at grid points lo..hi (0-based) of temperature iteff
    (1-based) before MAIN loop 401's floor: 'lines' (XLINOP + TAPE12 lines), 'xlinop' (XLINOP alone) and 'floor'
    (CONTINUUM = 1e-3 of the continuum); dfsynthe._run_temperature's steps, default Config (stock flavour)."""
    from atlas_torch.dfsynthe import odf_io
    from atlas_torch.dfsynthe.dfsynthe import Config
    from atlas_torch.dfsynthe.ftz import scope
    from atlas_torch.dfsynthe.lines.accumulate import LineAccumulator
    from atlas_torch.dfsynthe.lines.continuum import continuum, layer_setup
    from atlas_torch.dfsynthe.lines.selectlines import iter_tape12, select_tape19
    from atlas_torch.dfsynthe.lines.tables import LENBUFF, device_tables
    from atlas_torch.dfsynthe.lines.xlinop import XLinop
    from atlas_torch.dfsynthe.lines.f32 import F32
    d = EXT / "runs" / "stock" / "odf" / mix
    data = EXT / "BasicATLAS" / "data" / "dfsynthe_files"
    cfg = Config()
    fl, dev = cfg.flavor, cfg.device
    with scope(cfg.ftz):
        xt = odf_io.read_xnfpdf(d / "xnfpdf.dat", iteff)
        mx = odf_io.read_xnfpdfmax(d / "xnfpdfmax.dat", iteff)
        tape19 = select_tape19(data, mx.teff, mx.wledge, mx.xnfdopmax, fl)
        tabs = device_tables(dev, fl)
        lay = layer_setup(xt, fl, dev)
        cont = continuum(xt.wledge, lay["ablog"], fl)
        nl = cont.shape[0]
        pad = LineAccumulator.PAD
        bufpad = torch.zeros(nl * LENBUFF + 2 * pad, dtype=F32, device=dev)
        buf = bufpad[pad:pad + nl * LENBUFF]
        XLinop(tabs, lay, cont, fl).run(buf, tape19)
        xl = buf.view(nl, LENBUFF)[:, lo:hi + 1].double().numpy().copy()
        cols = ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")

        def t12():
            for _, t in iter_tape12(data, mx.teff, mx.wledge, mx.xnfdopmax, fl, cfg.select_chunk, {}):
                yield {k: torch.from_numpy(getattr(t, k)) for k in cols}
        acc = LineAccumulator(tabs, lay["xne"], lay["txnxn"], lay["dopple"], lay["xnfdop"], cont,
                              max_elements=cfg.max_elements, fastmath=fl.fastmath)
        acc.accumulate(buf, t12, chunk=cfg.line_chunk, padded=bufpad)
        b = buf.view(nl, LENBUFF)[:, lo:hi + 1].double().numpy().copy()
        return {"lines": b, "xlinop": xl, "floor": cont[:, lo:hi + 1].double().numpy().copy()}
