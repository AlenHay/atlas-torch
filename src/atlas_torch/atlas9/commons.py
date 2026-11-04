"""Canonical COMMON layout of ATLAS9 (names as declared in the main program) and the initial State.

The initial State is what the Fortran has at program start: zeros (COMMON in .bss, -finit-local-zero) overlaid
with the BLOCK DATA-like subroutines' DATA (BLOCKR, BLOCKE, BLOCKBIG, BLOCKJ, BLOCKH) and the COMMON members
DATA-initialised in ROSSTAB (/TABTP/ grid). Keys are "block.var" as in the golden traces; arrays keep the
Fortran declared shapes (layer index first, index 0 = Fortran 1).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import torch

from ..state import State
from . import data

KW = 99

# block: members. Type: implicit (I-N integer, else REAL*8) unless suffixed  :c<n> (CHARACTER*n), :f4 (REAL*4),
# :r (REAL*8 despite I-N name), :i (INTEGER despite A-H/O-Z name).
_LAYOUT = """
abross: abross(99) tauros(99)
abtot: abtot(99) alpha(99)
biglit: wavedf(1213,2)
cont: abtotc(99) alphac(99) taunuc(99) snuc(99) hnuc(99) jnuc(99):r jminsc(99):r residc(99)
conv: dltdlp(99) heatcp(99) dlrdlt(99) velsnd(99) grdadb(99) hscale(99) flxcnv(99) vconv(99) mixlth:r overwt
      flxcnv0(99) flxcnv1(99) ifconv nconv
depart: bhyd(99,6) bmin(99) nlteon
elem: abund(99) atmass(99) elem(99):c2
edens: edens(99) ifedns
filename: filename:c60
flux: flux flxerr(99) flxdrv(99) flxrad(99)
freq: freq freqlg ehvkt(99) stim(99) bnu(99)
freset: freset(1563) rcoset(1563) nulo nuhi numnu
height: height(99)
if: ifcorr ifpres ifsurf ifscat tauscat ifmol
ifop: ifop(20)
ions: xnfph(99,2) xnfphe(99,3) xnfh(99,2) xnfhe(99,3)
iter: iter ifprnt(15) ifpnch(15) numits
junk: xscale title(74):c1 freqid(6):c1 wlte:c4
mus: angle(20) surfi(20) nmu
ops: ahyd(99) ah2p(99) ahmin(99) sigh(99) ahe1(99) ahe2(99) ahemin(99) sighe(99) acool(99) aluke(99) ahot(99)
     sigel(99) sigh2(99) ahline(99) alines(99) siglin(99) axline(99) sigxl(99) axcont(99) sigx(99) shyd(99)
     shmin(99) shline(99) sxline(99) sxcont(99)
optot: acont(99) scont(99) aline(99) sline(99) sigmac(99) sigmal(99)
ptotal: ptotal(99)
put: put iput
pzero: pzero pcon pradk0 pturb0 knu(99):r pradk(99) raden(99)
rad: accrad(99) prad(99)
rhox: rhox(99) nrhox
state: p(99) xne(99) xnatom(99) rho(99)
taushj: taunu(99) snu(99) hnu(99) jnu(99):r jmins(99):r
steplg: steplg tau1lg krhox
teff: teff grav glog
temp: t(99) tkev(99) tk(99) hkt(99) tlog(99) hckt(99) itemp
turbpr: vturb(99) pturb(99) trbfdg trbcon trbpow trbsnd ifturb
wavey: wbegin deltaw ifwave
xabund: xabund(99) wtmole
xnf: xnfc(99,6) xnfn(99,6) xnfo(99,6) xnfne(99,6) xnfmg(99,6) xnfsi(99,6) xnfs(99,6) xnffe(99,5)
xnfp: xnfpc(99,4) xnfpn(99,5) xnfpo(99,6) xnfpne(99,6) xnfpal(99,1) xnfpmg(99,2) xnfpsi(99,2) xnfpca(99,2)
      xnfpfe(99,1) xnfpch(99) xnfpoh(99)
rr: rjminsnu(99) rdiagjnu(99)
tsmooth: j1smooth j2smooth wtjm1 wtj wtjp1 tsmooth(99)
tabtp: tabt(57) tabp(42) tabkap(57,42,5) pkap(42) tabv(5) pkapv(5,42) iftabk ktab(57,42,5) nt np
matxj: cj(2601):f4
matxh: ch(2601):f4
free: numcol letcol last more iffail maxpow word(6):c1
ifequa: ifequa(101) kcomps(450) locj(161) code(160):r equil(6,160) idequa(25) nequa nummol nloc
xnmol: xnmol(99,160)
xnsave: xnsave(99,25)
ifpop: ifpop
"""


@lru_cache(maxsize=None)
def layout() -> dict[str, list[tuple[str, tuple, str]]]:
    """block -> [(var, shape, dtype)] with numpy dtype strings ('f8', 'f4', 'i8' (ints widened), 'S<n>')."""
    out: dict[str, list] = {}
    blk = None
    for line in _LAYOUT.strip().splitlines():
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m:
            blk, rest = m.group(1), m.group(2)
            out[blk] = []
        else:
            rest = line
        for tok in rest.split():
            mm = re.match(r"^(\w+)(?:\(([\d,]+)\))?(?::(\w+))?$", tok)
            name, dims, typ = mm.group(1), mm.group(2), mm.group(3)
            shape = tuple(int(d) for d in dims.split(",")) if dims else (1,)   # scalars as (1,) like traces
            if typ and typ.startswith("c"):
                dt = f"S{int(typ[1:])}"
            elif typ == "f4":
                dt = "f4"
            elif typ == "r":
                dt = "f8"
            elif typ == "i":
                dt = "i8"
            else:
                dt = "i8" if name[0] in "ijklmn" else "f8"
            out[blk].append((name, shape, dt))
    return out


def _zeros(shape, dt):
    if dt.startswith("S"):
        return np.full(shape, b" " * int(dt[1:]), dtype=dt)
    if dt == "f4":
        return torch.zeros(shape, dtype=torch.float32)
    if dt == "i8":
        return torch.zeros(shape, dtype=torch.int64)
    return torch.zeros(shape, dtype=torch.float64)


def _put(s: State, key: str, value):
    cur = s[key]
    if isinstance(cur, np.ndarray):
        s[key] = np.asarray(value, dtype=cur.dtype).reshape(cur.shape)
    else:
        v = torch.as_tensor(np.asarray(value), dtype=cur.dtype)
        s[key] = v.reshape(cur.shape).clone()


def initial_state() -> State:
    """State at program start (before READIN)."""
    s = State()
    for blk, members in layout().items():
        for name, shape, dt in members:
            s[f"{blk}.{name}"] = _zeros(shape, dt)
    # CHARACTER COMMON members start as NUL bytes in the Fortran (.bss), not blanks
    for blk, members in layout().items():
        for name, shape, dt in members:
            if dt.startswith("S"):
                s[f"{blk}.{name}"] = np.full(shape, b"", dtype=dt)
    t = data.table
    # BLOCKR
    for k in ("dltdlp", "heatcp", "dlrdlt", "velsnd", "grdadb", "hscale", "flxcnv", "vconv", "mixlth", "overwt",
              "ifconv", "nconv"):
        _put(s, f"conv.{k}", t("blockr", k))
    _put(s, "depart.bhyd", t("blockr", "bhyd"))
    _put(s, "depart.bmin", t("blockr", "bmin"))
    _put(s, "depart.nlteon", t("blockr", "nlteon"))
    _put(s, "flux.flux", t("blockr", "flux"))
    _put(s, "flux.flxerr", t("blockr", "flxerr"))
    _put(s, "flux.flxdrv", t("blockr", "flxdrv"))
    _put(s, "freset.numnu", t("blockr", "numnu"))
    _put(s, "freset.freset", data.freset_init())
    for k in ("ifcorr", "ifpres", "ifsurf", "ifscat", "tauscat", "ifmol"):
        _put(s, f"if.{k}", t("blockr", k))
    _put(s, "ifop.ifop", t("blockr", "ifop"))
    for k in ("xnfph", "xnfphe"):
        _put(s, f"ions.{k}", t("blockr", k))
    for k in ("ifprnt", "ifpnch", "numits"):
        _put(s, f"iter.{k}", t("blockr", k))
    _put(s, "junk.title", t("blockr", "title"))
    _put(s, "junk.wlte", t("blockr", "wlte"))
    _put(s, "junk.xscale", t("blockr", "xscale"))
    _put(s, "mus.surfi", t("blockr", "surfi"))
    _put(s, "mus.nmu", t("blockr", "nmu"))
    for k in ("accrad", "prad"):
        _put(s, f"rad.{k}", t("blockr", k))
    for k in ("steplg", "tau1lg", "krhox"):
        _put(s, f"steplg.{k}", t("blockr", k))
    _put(s, "rhox.nrhox", t("blockr", "nrhox"))
    for k in ("teff", "grav"):
        _put(s, f"teff.{k}", t("blockr", k))
    for k in ("vturb", "pturb", "ifturb", "trbfdg", "trbpow", "trbsnd", "trbcon"):
        _put(s, f"turbpr.{k}", t("blockr", k))
    _put(s, "wavey.ifwave", t("blockr", "ifwave"))
    for k in ("j1smooth", "j2smooth", "wtjm1", "wtj", "wtjp1"):
        _put(s, f"tsmooth.{k}", t("blockr", k))
    # BLOCKE
    for k in ("abund", "atmass", "elem"):
        _put(s, f"elem.{k}", t("blocke", k))
    # BLOCKBIG, BLOCKJ, BLOCKH
    _put(s, "biglit.wavedf", data.wavedf())
    jt = data.josh_tables()
    _put(s, "matxj.cj", jt["coefj"].ravel(order="F"))
    _put(s, "matxh.ch", jt["coefh"].ravel(order="F"))
    # ROSSTAB's DATA for /TABTP/ (the KTAB tables are only used if READ KAPPA is absent; see rosstab.py)
    g = data.rosstab_grid()
    for k in ("tabt", "tabp", "tabv"):
        _put(s, f"tabtp.{k}", g[k])
    ktab = np.zeros((57, 42, 5), dtype=np.int64)
    for iv, v in enumerate((0, 1, 2, 4, 8)):
        for ip in range(42):
            ktab[:, ip, iv] = t("rosstab", f"k{v}tab{ip + 1:02d}")
    _put(s, "tabtp.ktab", ktab)
    _put(s, "tabtp.nt", 57)
    _put(s, "tabtp.np", 42)
    return s


def canonical_keys() -> list[str]:
    return [f"{b}.{n}" for b, ms in layout().items() for n, _, _ in ms]
