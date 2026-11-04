"""Helium bound-free and free-free absorption (Kurucz 1970 secs. 5.5-5.7; atlas7v.for HE1OP, HE2OP, HEMIOP).
Per unit volume, cm^-1, (..., L, F)."""
from __future__ import annotations

import torch

from .hydrogenic import free_free, merged_levels
from .state import C_CM, Conditions, Levels, stack

HE1_LIMIT, R_HE = 198310.76, 109722.267          # He I ionisation limit, Rydberg constant of He (cm^-1)
HE2_LIMIT, R_HE2 = 438908.85, 438889.068         # He II limit, 4 R_He

# He I levels below n = 6 (HE1OP): edge (cm^-1), g, excitation energy (cm^-1), cross-section form
#   ("kramers", n): 2.815E29/(n^5 nu^3); ("exp", a, b): exp(a - b ln nu); "2s3S", "1s1S": the fits of sec. 5.5
_HE1 = (
    (4368.190, 3.0, 193942.57, ("kramers", 5)), (4388.260, 9.0, 193922.5, ("kramers", 5)),
    (4388.260, 27.0, 193922.5, ("kramers", 5)), (4389.390, 7.0, 193921.37, ("kramers", 5)),
    (4389.450, 15.0, 193921.31, ("kramers", 5)), (4392.369, 5.0, 193918.391, ("kramers", 5)),
    (4393.515, 15.0, 193917.245, ("kramers", 5)), (4509.980, 9.0, 193800.78, ("kramers", 5)),
    (4647.133, 1.0, 193663.627, ("kramers", 5)), (4963.671, 3.0, 193347.089, ("kramers", 5)),
    (6817.943, 3.0, 191492.817, ("kramers", 4)), (6858.680, 7.0, 191452.08, ("kramers", 4)),
    (6858.960, 21.0, 191451.80, ("kramers", 4)), (6864.201, 5.0, 191446.559, ("kramers", 4)),
    (6866.172, 15.0, 191444.588, ("kramers", 4)), (7093.620, 9.0, 191217.14, ("kramers", 4)),
    (7370.429, 1.0, 190940.331, ("kramers", 4)), (8012.550, 3.0, 190298.210, ("kramers", 4)),
    (12101.289, 3.0, 186209.471, ("exp", 58.81, 2.89)), (12205.695, 5.0, 186105.065, ("exp", 85.20, 3.69)),
    (12209.106, 15.0, 186101.654, ("exp", 85.20, 3.69)), (12746.066, 9.0, 185564.694, ("exp", 49.30, 2.60)),
    (13445.824, 1.0, 184864.936, ("exp", 23.85, 1.86)), (15073.868, 3.0, 183236.892, ("exp", 12.69, 1.54)),
    (27175.760, 3.0, 171135.000, ("exp", 81.35, 3.5)), (29223.753, 9.0, 169087.007, ("exp", 61.21, 2.9)),
    (32033.214, 1.0, 166277.546, ("exp", 26.83, 1.91)), (38454.691, 3.0, 159856.069, "2s3S"),
    (198310.760, 1.0, 0.0, "1s1S"),
)

# He II levels n = 1..9 (HE2OP): edge (cm^-1), g = 2n^2, energy (cm^-1), Gaunt factor fit A + B/nu + C/nu^2 of
# hydrogen (Kurucz 1970 p. 75; None: g = 1)
_HE2 = (
    (438908.850, 1, 0.0, (0.9916, 2.719e13, -2.268e30)), (109726.529, 2, 329182.321, (1.105, -2.375e14, 4.077e28)),
    (48766.491, 3, 390142.359, (1.101, -9.863e13, 1.035e28)), (27430.925, 4, 411477.925, (1.101, -5.765e13, 4.593e27)),
    (17555.715, 5, 421353.135, (1.102, -3.909e13, 2.371e27)), (12191.437, 6, 426717.413, (1.0986, -2.704e13, 1.229e27)),
    (8956.950, 7, 429951.90, None), (6857.660, 8, 432051.19, None), (5418.390, 9, 433490.46, None),
)

HE1_FF_TYPO = 3.619e8      # HE1OP's free-free constant


def _he1_sigma(c: Conditions):
    def sigma(nu):
        ln = torch.log(nu)
        cols = []
        for _, _, _, form in _HE1:
            if form == "2s3S":
                cols.append(torch.exp(-390.026 + (21.035 - 0.318 * ln) * ln))
            elif form == "1s1S":
                cols.append(torch.exp(33.32 - 2.0 * ln))
            elif form[0] == "kramers":
                cols.append(c.rad.kramers / (form[1] ** 5 * nu ** 3))
            else:
                cols.append(torch.exp(form[1] - form[2] * ln))
        return stack(*cols)
    return Levels(tuple(r[1] for r in _HE1), tuple(r[2] for r in _HE1), tuple(r[0] * C_CM for r in _HE1), sigma)


def he1_bound_free(c: Conditions) -> torch.Tensor:
    """He I photoionisation: the 29 terms below n = 6 (n = 1, 2 after Gingerich 1964, n = 3 Hunger & van Blerkom 1967,
    n = 4, 5 hydrogenic) and n >= 6 hydrogenic with g = 4 n^2 by eq. 5.6 (sec. 5.5; HE1OP)."""
    total = _he1_sigma(c).weighted_sum(c) + merged_levels(c, 6, HE1_LIMIT, R_HE, 4.0)
    return c.n_over_u(2, 1) * total * c.stim


def he1_free_free(c: Conditions) -> torch.Tensor:
    """Free-free of electrons on He II ('He I free-free'), hydrogenic with z = 1 (sec. 5.5; HE1OP)."""
    return free_free(c, 1.0, c.n(2, 2), HE1_FF_TYPO if c.compat.he1_ff_typo else None)


def he2_bound_free(c: Conditions) -> torch.Tensor:
    """He II photoionisation: hydrogenic, z = 2, levels n = 1..6 with the hydrogen Gaunt-factor fits of Kurucz 1970
    p. 75 at nu/z^2, n = 7..9 Kramers, n >= 10 by eq. 5.6 (sec. 5.6; HE2OP)."""
    scale = 1.0 if c.compat.he2_gaunt_at_nu else 4.0

    def sigma(nu):
        cols = []
        for _, n, _, fit in _HE2:
            g = 1.0 if fit is None else fit[0] + (fit[1] + fit[2] / (nu / scale)) / (nu / scale)
            cols.append(c.rad.kramers * 16.0 / (n ** 5 * nu ** 3) * g)
        return stack(*cols)

    lv = Levels(tuple(2.0 * r[1] ** 2 for r in _HE2), tuple(r[2] for r in _HE2), tuple(r[0] * C_CM for r in _HE2),
                sigma)
    total = lv.weighted_sum(c) + merged_levels(c, 10, HE2_LIMIT, R_HE2, 2.0, z=2)
    return c.n_over_u(2, 2) * total * c.stim


def he2_free_free(c: Conditions) -> torch.Tensor:
    """Free-free of electrons on He III ('He II free-free'), hydrogenic z = 2 (sec. 5.6; HE2OP)."""
    return free_free(c, 2.0, c.n(2, 3))


def he_minus_free_free(c: Conditions) -> torch.Tensor:
    """He- free-free (He I + e), John (1968) fit by Carbon, Gingerich & Latham (1969): n(He I, 1s) n_e (aT + b + c/T),
    stimulated emission included (sec. 5.7; HEMIOP); n(He I, 1s) = n/U (g = 1)."""
    f = c.nu
    a = 3.397e-46 + (-5.216e-31 + 7.039e-15 / f) / f
    b = -4.116e-42 + (1.067e-26 + 8.135e-11 / f) / f
    cc = 5.081e-37 + (-8.724e-23 - 5.659e-8 / f) / f
    return (a * c.T + b + cc / c.T) * c.n_e * c.n_over_u(2, 1)
