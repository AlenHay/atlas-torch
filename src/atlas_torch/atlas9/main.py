"""PROGRAM ATLAS9 (atlas9mem.for as patched by BasicATLAS): READIN / iterate / READIN ... until a STOP or the
end of the control cards.

One iteration: hydrostatic pressure, populations (Physics), PUTOUT(1), TCORR/ROSS/RADIAP(1), the frequency
integration, ROSS(3), HIGH, RADIAP(3), CONVEC, TCORR(3), TURB, PUTOUT(5).

Frequency integration (IFOP(15)=1 ODF line path, IFSURF=0) -- batched instead of the Fortran's NU loop:
  1. FREQ, BNU, STIM, EHVKT for all NU ([F, n]);
  2. continuum opacities for all NU (Physics.continuum) and KAPP's combination;
  3. LINOP for all NU and all 12 sub-steps (the ODF intervals replayed with LINOP's forward-only search);
  4. JOSH for the F continuum problems and the 12F line problems in one batch (they are independent);
  5. the opacity-sampling control flow of the main program, replayed per NU on scalars: step N is used
     until RESIDC(1)=HNU(1)/HNUC(1) > .9995 or all RESIDC(J) >= .998 (REAL*4 literals), then the remaining
     weight 1-SUMWT (0 if < .0001) goes to the continuum solution -- whose KNU(1) is not restored (RADIAP
     uses the last line step's KNU(1));
  6. the TCORR/RADIAP/ROSS(2) increments of all used (NU, step) contributions, accumulated in the Fortran's
     order with sequential float64 sums (cumsum).
The SAVEd/COMMON leftovers of the loop (/FREQ/, /TAUSHJ/, /ABTOT/, /CONT/, /OPTOT/, /OPS/, /RR/, KNU(1)) are
set to their values after the last NU, as in the Fortran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..fnum import f32
from ..state import State
from . import fmath, linop, putout
from .commons import initial_state
from .convec import convec
from .errors import FortranExit
from .high import high
from .josh import josh
from .kapp import OPS, kapp_combine
from .radiap import radiap3, radiap_terms
from .readin import IO, readin
from .ross import ross3, ross_terms
from .tcorr import tcorr1, tcorr3, tcorr_terms
from .turb import turb
from .cards import StdinReader

_R9995, _R998, _R0001 = f32(0.9995), f32(0.998), f32(0.0001)


@dataclass
class RunResult:
    stdout: list
    punch: list
    exit_message: str
    exit_code: int
    states: list = field(default_factory=list)       # State after each PUTOUT(5)


def _vec(s: State, key: str, n: int) -> torch.Tensor:
    return s[key][:n]


def hydrostatic(s: State) -> dict:
    """Main program loop 11 (IFPRES=1): PZERO, PTOTAL, P (with the PRAD(1) / P(J-1) fall-backs)."""
    n = s.iscalar("rhox.nrhox")
    pcon, pradk0, pturb0 = s.scalar("pzero.pcon"), s.scalar("pzero.pradk0"), s.scalar("pzero.pturb0")
    pzero = pcon + pradk0 + pturb0
    grav = s.scalar("teff.grav")
    rx, prad, pturb = _vec(s, "rhox.rhox", n), _vec(s, "rad.prad", n), _vec(s, "turbpr.pturb", n)
    ptotal = s["ptotal.ptotal"].clone(); ptotal[:n] = grav * rx + pzero
    p = grav * rx - prad - pturb - pcon
    bad = ~(p > 0)
    if bool(bad.any()):
        p2 = grav * rx - (prad - prad[0]) - pturb - pcon
        p = torch.where(bad, p2, p)
        for j in torch.nonzero(~(p > 0)).flatten().tolist():   # p(j)=p(j-1) for p<0 (sequential, rare)
            if p[j] < 0.0:
                p[j] = p[j - 1]
    pp = s["state.p"].clone(); pp[:n] = p
    return {"pzero.pzero": torch.tensor([pzero]), "ptotal.ptotal": ptotal, "state.p": pp}


def frequencies(s: State) -> dict:
    """FREQ, RCO, FREQLG, EHVKT, STIM, BNU for NU=NULO..NUHI (loop 20)."""
    n = s.iscalar("rhox.nrhox")
    nulo, nuhi = s.iscalar("freset.nulo"), s.iscalar("freset.nuhi")
    nus = torch.arange(nulo, nuhi + 1)
    if s.iscalar("wavey.ifwave") == 0:
        freq = s["freset.freset"][nus - 1].clone()
        rco = s["freset.rcoset"][nus - 1].clone()
    else:
        wb, dw = s.scalar("wavey.wbegin"), s.scalar("wavey.deltaw")
        k = (nus - nulo).to(torch.float64)
        if wb > 1.0e10:
            freq = wb + k * dw
            rco = torch.full_like(freq, dw)
        else:
            wave = wb + k * dw
            freq = fmath.rdiv(2.997925e17, wave)
            rco = torch.abs(dw / wave * freq)
    hkt = _vec(s, "temp.hkt", n)
    ehvkt = fmath.exp(-freq.unsqueeze(1) * hkt.unsqueeze(0))
    stim = 1.0 - ehvkt
    f15 = freq / 1.0e15
    bnu = 1.47439e-2 * (f15 * f15 * f15).unsqueeze(1) * ehvkt / stim
    return {"nu": nus, "freq": freq, "rco": rco, "freqlg": fmath.log(freq), "ehvkt": ehvkt, "stim": stim, "bnu": bnu}


def frequency_integration(s: State, physics, odf: torch.Tensor, stdout: list, punch: list) -> dict:
    n = s.iscalar("rhox.nrhox")
    if int(s["ifop.ifop"][14]) != 1 or int(s["ifop.ifop"][15]) == 1:
        raise NotImplementedError("only the ODF line path (IFOP(15)=1, IFOP(16)=0) is ported")
    if s.iscalar("if.ifsurf") != 0:
        raise NotImplementedError("IFSURF>0 (surface flux / intensity runs) is not ported")
    fq = frequencies(s)
    F = fq["freq"].shape[0]
    ops = physics.continuum(s, fq)
    zero = torch.zeros(F, n, dtype=torch.float64)
    ops = {k: ops.get(k, zero) for k in OPS}
    cont = kapp_combine(ops, fq["bnu"])
    # LINOP for all NU, 12 steps
    waves = (2.997925e17 / fq["freq"]).tolist()
    iwls = linop.interval_index(waves)
    it, ip, co, fscat = linop.coefficients(_vec(s, "temp.tlog", n), _vec(s, "state.p", n),
                                           _vec(s, "abross.tauros", n), s.scalar("if.tauscat"))
    has = torch.tensor([w is not None for w in iwls])
    iwl_t = torch.tensor([w if w is not None else 1 for w in iwls])
    alines, siglin = linop.line_opacity(odf, iwl_t, it, ip, co, fscat)          # [F, 12, n]
    alines = torch.where(has[:, None, None], alines, torch.zeros_like(alines))
    # JOSH: F continuum problems + 12F line problems (ALINE=ALINES; SLINE, SIGMAL, ACONT... from KAPP(0))
    rep = lambda a: a.unsqueeze(1).expand(F, 12, n).reshape(F * 12, n)
    rhox = _vec(s, "rhox.rhox", n)
    ifscat, ifsurf = s.iscalar("if.ifscat"), s.iscalar("if.ifsurf")
    cat = lambda c, l: torch.cat([c, l], 0)
    jo = josh(cat(cont["acont"], rep(cont["acont"])), cat(cont["scont"], rep(cont["scont"])),
              cat(cont["aline"], alines.reshape(F * 12, n)), cat(cont["sline"], rep(cont["sline"])),
              cat(cont["sigmac"], rep(cont["sigmac"])), cat(cont["sigmal"], rep(cont["sigmal"])),
              rhox, ifscat, ifsurf)
    jc = {k: v[:F] for k, v in jo.items()}
    jl = {k: v[F:].reshape((F, 12) + v.shape[1:]) for k, v in jo.items()}
    # RESIDC after each line step
    hnuc = jc["hnu"].unsqueeze(1)
    resid = torch.where(hnuc > 0, jl["hnu"] / hnuc, torch.ones_like(jl["hnu"]))
    stop_r1 = (resid[..., 0] > _R9995).tolist()
    stop_all = (~(resid < _R998).any(-1)).tolist()
    # replay the control flow
    contrib = []          # (nu index, step (0..11) or -1 for the continuum remainder, stepwt, knu step)
    hsurf_l, contin_l, nsteps_l = [], [], []
    hnu1_l = jl["hnu"][..., 0].tolist()
    hnu1_c = jc["hnu"][:, 0].tolist()
    for f in range(F):
        nsteps = 12 if iwls[f] is not None else 1
        sumwt = 0.0
        hsurf = 0.0
        nstep = 1
        last = 0
        while True:
            stepwt = float(linop.WT[nstep - 1]) if nsteps == 12 else 1.0
            contrib.append((f, nstep - 1, stepwt, nstep - 1))
            sumwt = sumwt + stepwt
            hsurf = hsurf + hnu1_l[f][nstep - 1] * stepwt
            last = nstep - 1
            if nstep < nsteps:
                nstep += 1
                if stop_r1[f][last] or stop_all[f][last]:
                    break
                continue
            break
        stepwt = 1.0 - sumwt
        if stepwt < _R0001:
            stepwt = 0.0
        if stepwt != 0.0:
            contrib.append((f, -1, stepwt, last))
        hsurf = hsurf + hnu1_c[f] * stepwt
        hsurf_l.append(hsurf); contin_l.append(hnu1_c[f]); nsteps_l.append(nsteps)
    fi = torch.tensor([c[0] for c in contrib])
    st = torch.tensor([c[1] for c in contrib])
    wt = torch.tensor([c[2] for c in contrib], dtype=torch.float64)
    ks = torch.tensor([c[3] for c in contrib])
    isc = st < 0
    stc = torch.clamp(st, min=0)

    def pick(name):
        a = jl[name][fi, stc]
        b = jc[name][fi]
        if a.dim() == 2:
            return torch.where(isc[:, None], b, a)
        return torch.where(isc, b, a)

    abtot, alpha, taunu, hnu, jnu, jmins = (pick(k) for k in ("abtot", "alpha", "taunu", "hnu", "jnu", "jmins"))
    knu1 = jl["knu1"][fi, ks]
    rcowt = fq["rco"][fi] * wt
    bnu, stim, freq = fq["bnu"][fi], fq["stim"][fi], fq["freq"][fi]
    t, hkt = _vec(s, "temp.t", n), _vec(s, "temp.hkt", n)
    numnu = s.iscalar("freset.numnu")
    out = {}
    seq = lambda inc: torch.cumsum(inc, dim=0)[-1]
    if s.iscalar("if.ifcorr") == 1:
        tc = tcorr_terms(rhox, abtot, hnu, jmins, taunu, alpha, bnu, freq, hkt, t, stim, rcowt,
                         s.scalar("teff.teff"), numnu, s.scalar("flux.flux"))
        for k, key in (("rdabh", "save.tcorr.rdabh"), ("rjminsnu", "save.tcorr.rjmins"),
                       ("rdiagjnu", "save.tcorr.rdiagj"), ("flxrad", "flux.flxrad")):
            a = s[key].clone(); a[:n] = seq(torch.cat([a[:n].unsqueeze(0), tc[k]], 0)); out[key] = a
        for k, key in (("rjminsnu", "rr.rjminsnu"), ("rdiagjnu", "rr.rdiagjnu")):
            a = s[key].clone(); a[:n] = tc[k][-1]; out[key] = a
    rt = radiap_terms(jnu, hnu, abtot, knu1, rcowt)
    for k, key in (("raden", "pzero.raden"), ("h", "save.radiap.h"), ("accrad", "rad.accrad")):
        a = s[key].clone(); a[:n] = seq(torch.cat([a[:n].unsqueeze(0), rt[k]], 0)); out[key] = a
    pk = seq(torch.cat([s["pzero.pradk0"].to(torch.float64), rt["pradk0"]]))
    out["pzero.pradk0"] = pk.reshape(1)
    rs = ross_terms(bnu, freq, hkt, t, stim, abtot, rcowt, numnu)
    a = s["abross.abross"].clone(); a[:n] = seq(torch.cat([a[:n].unsqueeze(0), rs], 0)); out["abross.abross"] = a
    # leftovers of the last NU
    fl = F - 1
    def put(key, v):
        a = s[key].clone(); a[:n] = v; out[key] = a
    put("freq.ehvkt", fq["ehvkt"][fl]); put("freq.stim", fq["stim"][fl]); put("freq.bnu", fq["bnu"][fl])
    out["freq.freq"] = fq["freq"][fl].reshape(1).clone(); out["freq.freqlg"] = fq["freqlg"][fl].reshape(1).clone()
    for k, key in (("abtot", "cont.abtotc"), ("alpha", "cont.alphac"), ("taunu", "cont.taunuc"), ("snu", "cont.snuc"),
                   ("hnu", "cont.hnuc"), ("jnu", "cont.jnuc"), ("jmins", "cont.jminsc")):
        put(key, jc[k][fl])
    for k, key in (("abtot", "abtot.abtot"), ("alpha", "abtot.alpha"), ("taunu", "taushj.taunu"),
                   ("snu", "taushj.snu"), ("hnu", "taushj.hnu"), ("jnu", "taushj.jnu"), ("jmins", "taushj.jmins")):
        put(key, jc[k][fl])
    knu = s["pzero.knu"].clone(); knu[0] = jl["knu1"][fl, contrib[-1][3]]; out["pzero.knu"] = knu
    for k in OPS:
        put(f"ops.{k}", ops[k][fl])
    put("ops.alines", alines[fl, contrib[-1][3]] if iwls[fl] is not None else ops["alines"][fl])
    put("ops.siglin", siglin[fl, contrib[-1][3]] if iwls[fl] is not None else ops["siglin"][fl])
    for k in ("acont", "scont", "sline", "sigmac", "sigmal"):
        put(f"optot.{k}", cont[k][fl])
    put("optot.aline", alines[fl, contrib[-1][3]] if iwls[fl] is not None else cont["aline"][fl])
    rc = s["cont.residc"].clone(); rc[:n] = 1.0; out["cont.residc"] = rc
    # PUTOUT(2..4) output
    it_ = s.iscalar("iter.iter")
    ifprnt, ifpnch = int(s["iter.ifprnt"][it_ - 1]), int(s["iter.ifpnch"][it_ - 1])
    if ifprnt >= 3:
        raise NotImplementedError("PUTOUT per-frequency dumps for IFPRNT>=3 are not ported")
    if ifprnt > 1 or ifpnch >= 2:
        nus = fq["nu"].tolist()
        for f in range(F):
            if ifprnt > 1 and nsteps_l[f] > 1:
                stdout.append("0")
            so, pu = putout.mode4_lines(nus[f], float(fq["freq"][f]), hsurf_l[f], contin_l[f], nsteps_l[f],
                                        jc["taunu"][f], ifprnt, ifpnch, s.iscalar("freset.nuhi"), ifsurf)
            stdout.extend(so); punch.extend(pu)
    return out


def iteration(s: State, physics, odf, stdout: list, punch: list, hook=None) -> State:
    """One pass of DO 100 ITERAT (s has ITER and ITEMP already set). `hook(name, state)` is called before
    each step (names: populations, putout1, frequencies, ross3, high, radiap3, convec, tcorr3, turb, putout5,
    end) -- used by the tests to run the Fortran routines on the same inputs."""
    hook = hook or (lambda name, st: None)
    if s.iscalar("if.ifpres") != 0:
        s = s.updated(hydrostatic(s))
        s = s.updated({"edens.ifedns": torch.tensor([0])})
        s = s.updated(physics.electrons(s))
    hook("populations", s)
    s = s.updated(physics.populations(s))
    hook("putout1", s)
    it = s.iscalar("iter.iter")
    if int(s["iter.ifprnt"][it - 1]) != 0 and int(s["iter.ifpnch"][it - 1]) >= 2:
        punch.append("TEFF " + putout.ff.F(s.scalar("teff.teff"), 7, 0) + "  GRAVITY"
                     + putout.ff.F(s.scalar("teff.glog"), 8, 5) + " " + putout._str(s, "junk.wlte"))
        punch.append("TITLE " + putout._str(s, "junk.title"))
    if s.iscalar("if.ifcorr") == 1:
        s = s.updated(tcorr1(s))
    n = s.iscalar("rhox.nrhox")
    ab = s["abross.abross"].clone(); ab[:n] = 0.0
    h = s.get("save.radiap.h", torch.zeros(s["temp.t"].shape)).clone(); h[:n] = 0.0
    rd = s["pzero.raden"].clone(); rd[:n] = 0.0
    ac = s["rad.accrad"].clone(); ac[:n] = 0.0
    s = s.updated({"abross.abross": ab, "save.radiap.h": h, "pzero.raden": rd, "rad.accrad": ac,
                   "pzero.pradk0": torch.tensor([0.0])})
    hook("frequencies", s)
    s = s.updated(frequency_integration(s, physics, odf, stdout, punch))
    hook("ross3", s)
    s = s.updated(ross3(s))
    hook("high", s)
    s = s.updated(high(s))
    hook("radiap3", s)
    s = s.updated(radiap3(s))
    if s.iscalar("if.ifpres") == 1 and s.iscalar("conv.ifconv") == 1:
        hook("convec", s)
        s = s.updated(convec(s, physics))
    if s.iscalar("if.ifcorr") == 1:
        hook("tcorr3", s)
        s = s.updated(tcorr3(s))
    if s.iscalar("depart.nlteon") == 1:
        raise NotImplementedError("STATEQ (NLTE) is not ported")
    if s.iscalar("turbpr.ifturb") == 1:
        hook("turb", s)
        s = s.updated(turb(s))
    hook("putout5", s)
    s = s.updated(putout.mode5(s, stdout, punch))
    hook("end", s)
    return s


def run(stdin_text: str, files: dict, physics, max_iterations: int | None = None,
        on_iteration=None, hook=None, keep_states: bool = True) -> RunResult:
    """Run the program on control cards `stdin_text` with units {1: kappa.ros, 2: molecules, 3: punch deck,
    9: ODF}. `physics` implements Physics (see physics.py). Stops like the Fortran (STOP / CALL EXIT / end of
    the control cards = runtime error status 2), or after `max_iterations` iterations. keep_states=False
    does not keep the per-iteration States in the result (long runs)."""
    io = IO(stdin=StdinReader(stdin_text), files=files)
    s = initial_state()
    s["temp.itemp"] = torch.tensor([0])
    states = []
    odf = None
    count = 0
    try:
        while True:
            s = readin(s, io, 1)
            if odf is None and int(s["ifop.ifop"][14]) == 1:
                odf = linop.read_odf(str(files[9]))
            for iterat in range(1, s.iscalar("iter.numits") + 1):
                s = s.updated({"iter.iter": torch.tensor([iterat]),
                               "temp.itemp": s["temp.itemp"] + iterat})
                s = iteration(s, physics, odf, io.stdout, io.punch, hook)
                if keep_states:
                    states.append(s)
                count += 1
                if on_iteration is not None:
                    on_iteration(count, s)
                if max_iterations is not None and count >= max_iterations:
                    return RunResult(io.stdout, io.punch, "max_iterations", 0, states)
    except FortranExit as e:
        return RunResult(io.stdout, io.punch, e.message, e.code, states)
