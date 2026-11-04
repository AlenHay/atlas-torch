/* NMOLEC's per-layer Newton iteration (atlas9mem.for, labels 41-110), a literal transcription of the Python
 * reference loop in atlas_torch.atlas9.physics.molec.nmolec (which follows the Fortran):
 *   EQ/DEQ assembly (static part, then the molecule TERM scatter events in Fortran order), SOLVIT (Gaussian
 *   elimination with partial pivoting, the Fortran's sequence of row operations), the damped update with the
 *   *.69 sign-flip factor, the /SCALE limiter and its SQRT, the RATIO>1E-4 convergence test and the NCHEMITER cap.
 * IEEE double arithmetic only, no contraction (built with -ffp-contract=off); sqrt is correctly rounded.
 * Pivot search: first index of the maximum |A(K,I)| with NaN counting as maximal (numpy argmax semantics of the
 * reference implementation).
 */
#include <math.h>
#include <string.h>

/* SOLVIT(A,N,B): a is column-major N x N (a[i + j*n] = A(i+1,j+1)), b the right-hand side; both overwritten,
 * b returns the solution.  ipiv: N ints of scratch. */
static void solvit(double *a, double *b, int n, int *ipiv)
{
#define A_(i, j) a[(i) + (long)(j) * n]
    for (int i = 0; i < n - 1; i++) {
        int m = i;
        double best = fabs(A_(i, i));
        if (!isnan(best)) {
            for (int k = i + 1; k < n; k++) {
                double v = fabs(A_(k, i));
                if (isnan(v)) { m = k; break; }
                if (v > best) { best = v; m = k; }
            }
        }
        ipiv[i] = m;
        if (m != i) {
            for (int j = i + 1; j < n; j++) { double t = A_(i, j); A_(i, j) = A_(m, j); A_(m, j) = t; }
        }
        double pivot = 1.0 / A_(m, i);
        A_(m, i) = A_(i, i);
        A_(i, i) = pivot;
        for (int k = i + 1; k < n; k++) A_(k, i) = A_(k, i) * pivot;
        for (int j = i + 1; j < n; j++) {
            double c = A_(i, j);
            if (c == 0.0) continue;
            for (int k = i + 1; k < n; k++) A_(k, j) = A_(k, j) - A_(k, i) * c;
        }
    }
    A_(n - 1, n - 1) = 1.0 / A_(n - 1, n - 1);
    for (int i = 0; i < n - 1; i++) {
        int m = ipiv[i];
        if (m != i) { double t = b[m]; b[m] = b[i]; b[i] = t; }
        double c = b[i];
        for (int k = i + 1; k < n; k++) b[k] = b[k] - A_(k, i) * c;
    }
    int j1 = n - 1;
    for (int r = 0; r < n - 1; r++) {
        int j = j1;
        j1 -= 1;
        b[j] = b[j] * A_(j, j);
        double c = b[j];
        for (int k = 0; k <= j1; k++) b[k] = b[k] - A_(k, j) * c;
    }
    b[0] = b[0] * A_(0, 0);
#undef A_
}

/* TERM(JMOL) = EQUILJ(JMOL) * prod XN(K) (/ XN(NEQUA) for the 1/XNE components), in component order. */
void nmolec_terms(int nummol, int maxc, const long *comp_pad, const unsigned char *is_inv, const double *equilj,
                  const double *xn, double *term)
{
    for (int j = 0; j < nummol; j++) {
        double t = equilj[j];
        for (int c = 0; c < maxc; c++) {
            long k = comp_pad[(long)j * maxc + c];
            if (k < 0) continue;
            if (is_inv[(long)j * maxc + c]) t = t / xn[k]; else t = t * xn[k];
        }
        term[j] = t;
    }
}

/* One layer: iterate until converged (or the cap).  xn (N) in/out.  work: >= N + N*N + 2*N + nummol doubles,
 * iwork: >= N ints.  Returns the iteration count; *chemfail set when the cap stopped it. */
long nmolec_newton(int n, int nummol, int maxc, const long *comp_pad, const unsigned char *is_inv,
                   long nev, const long *tgt, const long *ejm, const long *ediv, const double *esg,
                   const double *xab, double xntot, int last_is_electron, const double *equilj,
                   double *xn, long cap, double r4_1e4, double r4_69, double *work, int *iwork, int *chemfail)
{
    double *flat = work;                  /* EQ(1..N), DEQ column-major */
    double *eqold = work + n + (long)n * n;
    double *sol = eqold + n;
    double *term = sol + n;
    double *eq = flat, *deq = flat + n;
    long it = 0;
    *chemfail = 0;
    for (int k = 0; k < n; k++) eqold[k] = 0.0;
    for (;;) {
        it++;
        memset(flat, 0, sizeof(double) * (n + (long)n * n));
        eq[0] = -xntot;
        for (int k = 1; k < n; k++) {
            eq[0] = eq[0] + xn[k];
            deq[0 + (long)k * n] = 1.0;
            eq[k] = xn[k] - xab[k] * xn[0];
            deq[k + (long)k * n] = 1.0;
            deq[k + 0] = -xab[k];
        }
        if (last_is_electron) {
            eq[n - 1] = -xn[n - 1];
            deq[(n - 1) + (long)(n - 1) * n] = -1.0;
        }
        nmolec_terms(nummol, maxc, comp_pad, is_inv, equilj, xn, term);
        for (long e = 0; e < nev; e++) {
            double v = term[ejm[e]];
            if (ediv[e] >= 0) v = v / xn[ediv[e]];
            v = v * esg[e];
            flat[tgt[e]] = flat[tgt[e]] + v;
        }
        for (int k = 0; k < n; k++) sol[k] = eq[k];
        solvit(deq, sol, n, iwork);
        int iferr = 0;
        double scale = 100.0;
        for (int k = 0; k < n; k++) {
            double e = sol[k];
            double ratio = fabs(e / xn[k]);
            if (ratio > r4_1e4) iferr = 1;
            if (eqold[k] * e < 0.0) e = e * r4_69;
            double xneq = xn[k] - e;
            double xn100 = xn[k] / 100.0;
            if (fabs(xneq) < xn100) {
                xn[k] = xn[k] / scale;
                if (eqold[k] * e < 0.0) scale = sqrt(scale);
            } else {
                xn[k] = fabs(xneq);
            }
            eqold[k] = e;
        }
        if (cap > 0 && it >= cap) { *chemfail = 1; break; }
        if (iferr != 1) break;
    }
    return it;
}
