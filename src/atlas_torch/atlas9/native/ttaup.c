/* TTAUP with ROSSTAB (atlas9mem.for), transcriptions of the Python/torch references
 * atlas_torch.atlas9.ttaup.ttaup and atlas_torch.atlas9.rosstab.Rosstab (MAP1 as numerics.map1_shared, one point):
 * IEEE double arithmetic in the reference's operation order, glibc log/log10/exp, no contraction.
 */
#include <math.h>

/* MAP1 at one point (NNEW=1): numerics.map1_shared / map1b arithmetic.  Interval search = torch.searchsorted
 * (right=True): first k with xold[k] > x (NaN -> nold). */
static double map1_point_L(const double *xold, int nold, const double *f, double x, int *Lout)
{
    int lo = 0, hi = nold;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (xold[mid] > x) hi = mid; else lo = mid + 1;
    }
    int L = lo + 1;
    *Lout = L;
    if (L < 2) L = 2;
    int lin = (L == 2) || (L == 3) || (L > nold);
    int last = (L == nold);
    int i = (L < nold ? L : nold) - 1;
#define CL(k) ((k) < 0 ? 0 : ((k) > nold - 1 ? nold - 1 : (k)))
    double xl = xold[CL(i)], xl1 = xold[CL(i - 1)], xl2 = xold[CL(i - 2)], xl3 = xold[CL(i + 1)];
    double fl = f[CL(i)], fl1 = f[CL(i - 1)], fl2 = f[CL(i - 2)], fl3 = f[CL(i + 1)];
#undef CL
    double d = (fl1 - fl2) / (xl1 - xl2);
    double cbac = fl / ((xl - xl1) * (xl - xl2)) + (fl2 / (xl - xl2) - fl1 / (xl - xl1)) / (xl1 - xl2);
    double bbac = d - (xl1 + xl2) * cbac;
    double abac = fl2 - xl2 * d + xl1 * xl2 * cbac;
    double d2 = (fl - fl1) / (xl - xl1);
    double cfor = fl3 / ((xl3 - xl) * (xl3 - xl1)) + (fl1 / (xl3 - xl1) - fl / (xl3 - xl)) / (xl - xl1);
    double bfor = d2 - (xl + xl1) * cfor;
    double afor = fl1 - xl1 * d2 + xl * xl1 * cfor;
    double wt = (fabs(cfor) != 0) ? fabs(cfor) / (fabs(cfor) + fabs(cbac)) : 0.0;
    double a = afor + wt * (abac - afor);
    double b = bfor + wt * (bbac - bfor);
    double c = cfor + wt * (cbac - cfor);
    if (last) { a = abac; b = bbac; c = cbac; }
    double bl = (fl - fl1) / (xl - xl1);
    double al = fl - xl * bl;
    if (lin) { a = al; b = bl; c = 0.0; }
    return a + (b + c * x) * x;
}

static double map1_point(const double *xold, int nold, const double *f, double x)
{
    int L;
    return map1_point_L(xold, nold, f, x, &L);
}

/* numerics.map1b for B rows (contiguous [B][nold], [B][nold], [B][nnew]): fnew [B][nnew], ll[B] = MAP1 = LL-1
 * with LL = min(L of the last point, NOLD). */
void map1b_rows(long B, int nold, int nnew, const double *xo, const double *fo, const double *xn, double *fnew,
                long *ll)
{
    for (long b = 0; b < B; b++) {
        int L = 0;
        for (int k = 0; k < nnew; k++)
            fnew[b * nnew + k] = map1_point_L(xo + b * nold, nold, fo + b * nold, xn[b * nnew + k], &L);
        if (L < 2) L = 2;
        ll[b] = (L < nold ? L : nold) - 1;
    }
}

/* ROSSTAB(T,P,V): tk[(ip*5+iv)*nt + it] = TABKAP(it,ip,iv); work >= 6*np doubles. */
double rosstab_point(const double *tabt, int nt, const double *tabp, int np, const double *tabv, const double *tk,
                     double ln10f, double t, double p, double v, double *work)
{
    double *pkapv = work, *pkap = work + 5 * np;
    double tlog10 = log10(t);
    for (int r = 0; r < 5 * np; r++) pkapv[r] = map1_point(tabt, nt, tk + (long)r * nt, tlog10);
    for (int ip = 0; ip < np; ip++) pkap[ip] = map1_point(tabv, 5, pkapv + 5 * ip, v);
    double plog = log10(p);
    if (!isnan(plog) && !(plog > tabp[0])) plog = tabp[0];     /* torch.maximum(plog, TABP(1)) */
    double ablog = map1_point(tabp, np, pkap, plog);
    return exp(ablog * ln10f);
}

/* TTAUP.  Returns 0, 1 (HYDROFAIL: N>1000), 2 (log of a non-positive number: Python's math.log raises),
 * 3 (exp overflow: math.exp raises). ab[0] is read when iross==1. */
int ttaup_c(const double *T, const double *TAU, const double *PRAD, const double *PTURB, const double *VT, double *AB,
            double *PT, double *P, double grav, int numtau, int iross, double tol, double p1,
            const double *tabt, int nt, const double *tabp, int np, const double *tabv, const double *tk, double ln10f,
            double *work)
{
#define LOG(dst, x) do { double _a = (x); if (_a <= 0.0) return 2; (dst) = log(_a); } while (0)
#define EXP(dst, x) do { double _a = (x); double _r = exp(_a); if (isinf(_r) && !isinf(_a)) return 3; (dst) = _r; } while (0)
    double dlgtau;
    LOG(dlgtau, TAU[1] / TAU[0]);
    double plog3 = 0.0, plog2 = 0.0, plog1 = 0.0, dplog2 = 0.0, dplog1 = 0.0, plog4 = 0.0, dplog3 = 0.0;
    double plog = 0.0, dplog = 0.0;
    if (iross != 1) {
        AB[0] = p1;
        if (PRAD[0] > 0.0) {
            double q = grav * TAU[0] / PRAD[0] / 2.0;
            AB[0] = (q < 0.1) ? q : 0.1;                            /* Python min(0.1, q) */
        }
    }
    for (int j = 0; j < numtau; j++) {
        int J = j + 1;
        if (J == 1) LOG(plog, grav / AB[0] * TAU[0]);
        else if (J <= 4) plog = plog1 + dplog1;
        else plog = (3.0 * plog4 + 8.0 * dplog1 - 4.0 * dplog2 + 8.0 * dplog3) / 3.0;
        double error = 1.0;
        int n = 1, first = 1;
        for (;;) {
            if (!first) {
                double pnew;
                if (J == 1) LOG(pnew, grav / AB[0] * TAU[0]);
                else if (J <= 4) pnew = (plog + 2.0 * plog1 + dplog + dplog1) / 3.0;
                else pnew = (126.0 * plog1 - 14.0 * plog3 + 9.0 * plog4 + 42.0 * dplog + 108.0 * dplog1
                             - 54.0 * dplog2 + 24.0 * dplog3) / 121.0;
                error = fabs(pnew - plog);
                plog = (pnew + plog) / 2.0;
            }
            first = 0;
            EXP(PT[j], plog);
            P[j] = PT[j] + (PRAD[0] - PRAD[j]) - PTURB[j];
            if (P[j] <= 0.0) P[j] = P[j > 0 ? j - 1 : numtau - 1];   /* Python P[j-1] (P[-1] at J=1) */
            AB[j] = rosstab_point(tabt, nt, tabp, np, tabv, tk, ln10f, T[j], P[j], VT[j], work);
            dplog = grav / AB[j] * TAU[j] / PT[j] * dlgtau;
            n += 1;
            if (n > 1000) return 1;
            if (!(error > tol)) break;
        }
        plog4 = plog3; plog3 = plog2; plog2 = plog1; plog1 = plog;
        dplog3 = dplog2; dplog2 = dplog1; dplog1 = dplog;
    }
    return 0;
#undef LOG
#undef EXP
}
