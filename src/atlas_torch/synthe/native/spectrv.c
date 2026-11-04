/* SPECTRV point loop (spectrv.for DO 25, BasicATLAS patch) for a batch of wavelength points: continuum opacity
   (DO 2006, 10.**x by scalar glibc pow), Planck function (DO 20, EXP(-FREQ*HKT) by glibc libmvec _ZGVbN2v_exp on
   layer pairs + scalar exp for an odd last layer, as the stock gfortran -O3 build; or scalar exp everywhere), then
   CALL JOSH(IFSCAT,1) for the continuum and CALL JOSH(1,1) with the fort.9 line opacity.

   JOSH/INTEG/PARCOE/MAP1/DERIV (atlas7v.for) are literal transcriptions of the Fortran (REAL*4 where the Fortran
   is REAL*4, same operation order), only the IFSURF=1 paths that SPECTRV takes for a flux spectrum.  The torch
   port (atlas_torch.synthe.spectrv_prog / josh.py) is the reference implementation and the portable fallback;
   this kernel exists for speed and must stay bit-identical to it.  Build without fast-math or FMA contraction:
   python -m atlas_torch.synthe.native (or `make native`).  Points are independent: optional pthreads over points. */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>

#define NXTAU 51
#define KWMAX 99

extern __m128d _ZGVbN2v_exp(__m128d);

typedef struct {
    int nrhox, ifscat, libm, kw;
    const double *wave; const int64_t *iedge;
    const double *wledge, *halfedge, *deledge;     /* (NEDGE) */
    const double *contabs, *contscat;              /* (NEDGE-1, 3, NRHOX) */
    const double *hkt, *rhox, *bfudge, *fscat;     /* (NRHOX) */
    const float *asynth; int64_t astride;          /* point p: asynth[p*astride + j] */
    const float *coefj;                            /* COEFJ(K,M) at coefj[(K-1)*51 + (M-1)] */
    const float *ch;                               /* CH(51) */
    const double *xtau8;                           /* XTAU8(51) */
    double *surf, *line;                           /* outputs: HNU(1) of both JOSH calls */
    int *thick;                                    /* MAXJ=1 branch taken by either call (diagnostic) */
    int64_t p0, p1;
} job_t;

static void parcoe(const double *f, const double *x, double *a, double *b, double *c, int n) {
    c[0] = 0.;
    b[0] = (f[1] - f[0]) / (x[1] - x[0]);
    a[0] = f[0] - x[0] * b[0];
    int n1 = n - 1;                                /* 1-based N1 = n-1 -> index n1-1 */
    c[n - 1] = 0.;
    b[n - 1] = (f[n - 1] - f[n1 - 1]) / (x[n - 1] - x[n1 - 1]);
    a[n - 1] = f[n - 1] - x[n - 1] * b[n - 1];
    if (n == 2) return;
    for (int J = 2; J <= n1; J++) {
        int j = J - 1, j1 = J - 2;
        double d = (f[j] - f[j1]) / (x[j] - x[j1]);
        c[j] = f[j + 1] / ((x[j + 1] - x[j]) * (x[j + 1] - x[j1])) - f[j] / ((x[j] - x[j1]) * (x[j + 1] - x[j]))
               + f[j1] / ((x[j] - x[j1]) * (x[j + 1] - x[j1]));
        b[j] = d - (x[j] + x[j1]) * c[j];
        a[j] = f[j1] - x[j1] * d + x[j] * x[j1] * c[j];
    }
    c[1] = 0.;
    b[1] = (f[2] - f[1]) / (x[2] - x[1]);
    a[1] = f[1] - x[1] * b[1];
    c[2] = 0.;
    b[2] = (f[3] - f[2]) / (x[3] - x[2]);
    a[2] = f[2] - x[2] * b[2];
    for (int J = 2; J <= n1; J++) {
        int j = J - 1, j1 = J;
        if (c[j] == 0.) continue;
        double wt = fabs(c[j1]) / (fabs(c[j1]) + fabs(c[j]));
        a[j] = a[j1] + wt * (a[j] - a[j1]);
        b[j] = b[j1] + wt * (b[j] - b[j1]);
        c[j] = c[j1] + wt * (c[j] - c[j1]);
    }
    a[n1 - 1] = a[n - 1];
    b[n1 - 1] = b[n - 1];
    c[n1 - 1] = c[n - 1];
}

static void integ(const double *x, const double *f, double *fint, int n, double start) {
    double a[KWMAX], b[KWMAX], c[KWMAX];
    parcoe(f, x, a, b, c, n);
    fint[0] = start;
    for (int i = 0; i < n - 1; i++)
        fint[i + 1] = fint[i] + (a[i] + b[i] / 2. * (x[i + 1] + x[i])
                                 + c[i] / 3. * ((x[i + 1] + x[i]) * x[i + 1] + x[i] * x[i])) * (x[i + 1] - x[i]);
}

/* FUNCTION MAP1(XOLD,FOLD,NOLD,XNEW,FNEW,NNEW), 1-based arrays emulated with a -1 offset */
static int map1(const double *xold0, const double *fold0, int nold, const double *xnew0, double *fnew0, int nnew) {
    const double *XO = xold0 - 1, *FO = fold0 - 1, *XN = xnew0 - 1;
    double *FN = fnew0 - 1;
    int L = 2, LL = 0, L1, L2;
    double A = 0, B = 0, C = 0, D, ABAC = 0, BBAC = 0, CBAC = 0, AFOR = 0, BFOR = 0, CFOR = 0, WT;
    for (int K = 1; K <= nnew; K++) {
    l10:
        if (XN[K] < XO[L]) goto l20;
        L = L + 1;
        if (L > nold) goto l30;
        goto l10;
    l20:
        if (L == LL) goto l50;
        if (L == 2) goto l30;
        if (L == 3) goto l30;
        L1 = L - 1;
        if (L > LL + 1 || L == 3) goto l21;
        if (L > LL + 1 || L == 4) goto l21;
        CBAC = CFOR; BBAC = BFOR; ABAC = AFOR;
        if (L == nold) goto l22;
        goto l25;
    l21:
        L2 = L - 2;
        D = (FO[L1] - FO[L2]) / (XO[L1] - XO[L2]);
        CBAC = FO[L] / ((XO[L] - XO[L1]) * (XO[L] - XO[L2]))
               + (FO[L2] / (XO[L] - XO[L2]) - FO[L1] / (XO[L] - XO[L1])) / (XO[L1] - XO[L2]);
        BBAC = D - (XO[L1] + XO[L2]) * CBAC;
        ABAC = FO[L2] - XO[L2] * D + XO[L1] * XO[L2] * CBAC;
        if (L < nold) goto l25;
    l22:
        C = CBAC; B = BBAC; A = ABAC;
        LL = L;
        goto l50;
    l25:
        D = (FO[L] - FO[L1]) / (XO[L] - XO[L1]);
        CFOR = FO[L + 1] / ((XO[L + 1] - XO[L]) * (XO[L + 1] - XO[L1]))
               + (FO[L1] / (XO[L + 1] - XO[L1]) - FO[L] / (XO[L + 1] - XO[L])) / (XO[L] - XO[L1]);
        BFOR = D - (XO[L] + XO[L1]) * CFOR;
        AFOR = FO[L1] - XO[L1] * D + XO[L] * XO[L1] * CFOR;
        WT = 0.;
        if (fabs(CFOR) != 0.) WT = fabs(CFOR) / (fabs(CFOR) + fabs(CBAC));
        A = AFOR + WT * (ABAC - AFOR);
        B = BFOR + WT * (BBAC - BFOR);
        C = CFOR + WT * (CBAC - CFOR);
        LL = L;
        goto l50;
    l30:
        if (L == LL) goto l50;
        L = nold < L ? nold : L;
        C = 0.;
        B = (FO[L] - FO[L - 1]) / (XO[L] - XO[L - 1]);
        A = FO[L] - XO[L] * B;
        LL = L;
    l50:
        FN[K] = A + (B + C * XN[K]) * XN[K];
    }
    return LL - 1;
}

static void deriv(const double *x, const double *f, double *dfdx, int n) {
    dfdx[0] = (f[1] - f[0]) / (x[1] - x[0]);
    int n1 = n - 1;
    dfdx[n - 1] = (f[n - 1] - f[n1 - 1]) / (x[n - 1] - x[n1 - 1]);
    if (n == 2) return;
    double s = fabs(x[1] - x[0]) / (x[1] - x[0]);
    for (int j = 1; j < n - 1; j++) {
        double m = fabs(f[j - 1]);
        if (fabs(f[j]) > m) m = fabs(f[j]);
        if (fabs(f[j + 1]) > m) m = fabs(f[j + 1]);
        double scale = m / fabs(x[j]);
        if (scale == 0.) scale = 1.;
        double d1 = (f[j + 1] - f[j]) / (x[j + 1] - x[j]) / scale;
        double d = (f[j] - f[j - 1]) / (x[j] - x[j - 1]) / scale;
        double tan1 = d1 / (s * sqrt(1. + d1 * d1) + 1.);
        double tan = d / (s * sqrt(1. + d * d) + 1.);
        dfdx[j] = (tan1 + tan) / (1. - tan1 * tan) * scale;
    }
}

/* SUBROUTINE JOSH(IFSCAT, IFSURF=1), split so that the REAL*4 Gauss-Seidel iteration (DO 34) of several
   independent systems can run lane by lane in one vector loop (gs_lanes).  josh_setup does labels 10 .. 31 and
   returns 0 with *hnu1 = HNU(1) when the system is finished (IFSCAT=0 path, or the MAXJ=1 branch 401: *thick=1),
   or 1 with XS/XSBAR/XALPHA/DIAG ready for DO 34. */
typedef struct { float xs[NXTAU], xsbar[NXTAU], xalpha[NXTAU], diag[NXTAU]; } gs_t;

static int josh_setup(const job_t *jb, int ifscat, const double *acont, const double *scont, const double *aline,
                      const double *sline, const double *sigmac, const double *sigmal, gs_t *g, double *hnu1,
                      int *thick) {
    const int n = jb->nrhox;
    const double *rhox = jb->rhox, *xtau8 = jb->xtau8;
    const float *coefj = jb->coefj;
    double abtot[KWMAX], alpha[KWMAX], snubar[KWMAX], taunu[KWMAX];
    for (int j = 0; j < n; j++) {
        abtot[j] = acont[j] + aline[j] + sigmac[j] + sigmal[j];
        alpha[j] = (sigmac[j] + sigmal[j]) / abtot[j];
        snubar[j] = (acont[j] * scont[j] + aline[j] * sline[j]) / (acont[j] + aline[j]);
    }
    integ(rhox, abtot, taunu, n, abtot[0] * rhox[0]);
    *thick = 0;
    if (ifscat != 1) {                                       /* SNU = SNUBAR; MAP1; GO TO 60 */
        double xs8[NXTAU];
        float xh = 0.f;
        map1(taunu, snubar, n, xtau8, xs8, NXTAU);
        for (int m = 0; m < NXTAU; m++) xh = xh + jb->ch[m] * (float)xs8[m];
        *hnu1 = (double)xh;
        return 0;
    }
    if (taunu[0] > xtau8[NXTAU - 1]) {                       /* MAXJ=1 -> 401 */
        double snu[KWMAX], hnu[KWMAX], jmins[KWMAX];
        *thick = 1;
        for (int j = 0; j < n; j++) snu[j] = snubar[j];
        for (int it = 0; it < NXTAU; it++) {
            double error = 0.;
            deriv(taunu, snu, hnu, n);
            for (int j = 0; j < n; j++) hnu[j] = hnu[j] / 3.;
            deriv(taunu, hnu, jmins, n);
            for (int j = 0; j < n; j++) {
                double jnu = jmins[j] + snu[j];
                double snew = (1. - alpha[j]) * snubar[j] + alpha[j] * jnu;
                error = fabs(snew - snu[j]) / snew + error;
                snu[j] = snew;
            }
            if (error < (double)0.00001f) break;
        }
        *hnu1 = hnu[0];
        return 0;
    }
    double xsbar8[NXTAU], xalpha8[NXTAU];
    map1(taunu, snubar, n, xtau8, xsbar8, NXTAU);
    map1(taunu, alpha, n, xtau8, xalpha8, NXTAU);
    for (int l = 0; l < NXTAU; l++) {
        double xa = xalpha8[l] > 0. ? xalpha8[l] : 0.;
        xa = xa < 1. ? xa : 1.;
        float xalpha = (float)xa;
        float xsbar = (float)(xsbar8[l] > 1e-38 ? xsbar8[l] : 1e-38);
        if (xtau8[l] < taunu[0]) {
            xsbar = (float)(snubar[0] > 1e-38 ? snubar[0] : 1e-38);
            double a1 = alpha[0] > 0. ? alpha[0] : 0.;
            xalpha = (float)(a1 < 1. ? a1 : 1.);
        }
        g->xs[l] = xsbar;
        g->xalpha[l] = xalpha;
        g->diag[l] = 1.f - xalpha * coefj[l * NXTAU + l];
        g->xsbar[l] = (1.f - xalpha) * xsbar;
    }
    return 1;
}

/* DO 34 / DO 33 / DO 32 for LN systems at once (lane l = system l; unused lanes: active[l] = 0), then label 60
   HNU(1) = sum CH(M)*XS(M).  Every lane does exactly the scalar Fortran operations in the Fortran order; a lane
   stops changing after the sweep in which it converged (IFERR=0). */
#define LN 8
static void gs_lanes(const job_t *jb, gs_t *const *g, int nl, double *hnu1) {
    const float *coefj = jb->coefj, *ch = jb->ch;
    float X[NXTAU][LN], XA[NXTAU][LN], XB[NXTAU][LN], DG[NXTAU][LN];
    int active[LN];
    for (int l = 0; l < LN; l++) {
        active[l] = l < nl;
        for (int m = 0; m < NXTAU; m++) {
            if (l < nl) { X[m][l] = g[l]->xs[m]; XA[m][l] = g[l]->xalpha[m]; XB[m][l] = g[l]->xsbar[m];
                          DG[m][l] = g[l]->diag[m]; }
            else { X[m][l] = 1.f; XA[m][l] = 0.f; XB[m][l] = 1.f; DG[m][l] = 1.f; }
        }
    }
    for (int sweep = 0; sweep < NXTAU; sweep++) {
        int iferr[LN] = {0};
        for (int k = NXTAU - 1; k >= 0; k--) {
            const float *ck = coefj + k * NXTAU;
            float d[LN];
            for (int l = 0; l < LN; l++) d[l] = 0.f;
            for (int m = 0; m < NXTAU; m++) {
                float c = ck[m];
                for (int l = 0; l < LN; l++) d[l] = d[l] + c * X[m][l];
            }
            for (int l = 0; l < LN; l++) {
                float dl = (d[l] * XA[k][l] + XB[k][l] - X[k][l]) / DG[k][l];
                float errorx = fabsf(dl / X[k][l]);
                iferr[l] |= errorx > 0.00001f;
                float s = X[k][l] + dl;
                float v = (double)s > 1e-38 ? s : (float)1e-38;
                X[k][l] = active[l] ? v : X[k][l];
            }
        }
        int any = 0;
        for (int l = 0; l < LN; l++) { active[l] = active[l] && iferr[l]; any |= active[l]; }
        if (!any) break;
    }
    for (int l = 0; l < nl; l++) {
        float xh = 0.f;
        for (int m = 0; m < NXTAU; m++) xh = xh + ch[m] * X[m][l];
        hnu1[l] = (double)xh;
    }
}

static void exp_layers(const job_t *jb, const double *arg, double *out) {
    int n = jb->nrhox;
    if (jb->libm == 1) {                                     /* libmvec 2-lane pairs, scalar exp for the odd last */
        int nv = n / 2 * 2;
        for (int j = 0; j < nv; j += 2) _mm_storeu_pd(out + j, _ZGVbN2v_exp(_mm_loadu_pd(arg + j)));
        for (int j = nv; j < n; j++) out[j] = exp(arg[j]);
    } else {
        for (int j = 0; j < n; j++) out[j] = exp(arg[j]);
    }
}

typedef struct {
    double acont[KWMAX], sigmac[KWMAX], ehvkt[KWMAX], stim[KWMAX], bnu[KWMAX];
} pt_t;

static void point_setup(const job_t *jb, int64_t p, pt_t *P) {
    const int n = jb->nrhox;
    double arg[KWMAX];
    double wave = jb->wave[p];
    int64_t ie = jb->iedge[p];                               /* 0-based IEDGE */
    double wl_i = jb->wledge[ie], wl_i1 = jb->wledge[ie + 1], he = jb->halfedge[ie], de = jb->deledge[ie];
    double c1 = (wave - he) * (wave - wl_i1) / de;
    double c2 = (wl_i - wave) * (wave - wl_i1) * 2. / de;
    double c3 = (wave - wl_i) * (wave - he) / de;
    const double *ca = jb->contabs + ie * 3 * n, *cs = jb->contscat + ie * 3 * n;
    for (int j = 0; j < n; j++) {
        P->acont[j] = pow(10., c1 * ca[j] + c2 * ca[n + j] + c3 * ca[2 * n + j]);
        P->sigmac[j] = pow(10., c1 * cs[j] + c2 * cs[n + j] + c3 * cs[2 * n + j]);
    }
    double freq = 2.99792458e17 / wave;
    double freq15 = freq / 1.e15;
    double f3 = freq15 * freq15 * freq15;
    for (int j = 0; j < n; j++) arg[j] = -freq * jb->hkt[j];
    exp_layers(jb, arg, P->ehvkt);
    for (int j = 0; j < n; j++) {
        P->stim[j] = 1. - P->ehvkt[j];
        P->bnu[j] = 1.47439e-02 * f3 * P->ehvkt[j] / P->stim[j];
    }
}

#define GP (LN / 2)                                          /* points per group: 2 JOSH systems each */
static void *run_job(void *arg) {
    job_t *jb = (job_t *)arg;
    const int n = jb->nrhox;
    double zero[KWMAX], aline[KWMAX], sline[KWMAX], sigmal[KWMAX];
    memset(zero, 0, sizeof zero);
    pt_t P[GP];
    gs_t G[2 * GP];
    for (int64_t p0 = jb->p0; p0 < jb->p1; p0 += GP) {
        int np = (int)(jb->p1 - p0 < GP ? jb->p1 - p0 : GP);
        double res[2 * GP], hl[LN];
        int th[2 * GP], need[2 * GP];
        gs_t *lanes[LN];
        int owner[LN], nl = 0;
        for (int q = 0; q < np; q++) {
            int64_t p = p0 + q;
            point_setup(jb, p, &P[q]);
            need[2 * q] = josh_setup(jb, jb->ifscat, P[q].acont, P[q].bnu, zero, P[q].bnu, P[q].sigmac, zero,
                                     &G[2 * q], &res[2 * q], &th[2 * q]);
            const float *asy = jb->asynth + p * jb->astride;
            for (int j = 0; j < n; j++) {
                double a = (double)asy[j];
                aline[j] = a * (1. - jb->fscat[j]);
                sline[j] = P[q].bnu[j] * P[q].stim[j] / (jb->bfudge[j] - P[q].ehvkt[j]);
                sigmal[j] = a * jb->fscat[j];
            }
            need[2 * q + 1] = josh_setup(jb, 1, P[q].acont, P[q].bnu, aline, sline, P[q].sigmac, sigmal,
                                         &G[2 * q + 1], &res[2 * q + 1], &th[2 * q + 1]);
            for (int s = 2 * q; s < 2 * q + 2; s++)
                if (need[s]) { lanes[nl] = &G[s]; owner[nl] = s; nl++; }
        }
        if (nl) {
            gs_lanes(jb, lanes, nl, hl);
            for (int l = 0; l < nl; l++) res[owner[l]] = hl[l];
        }
        for (int q = 0; q < np; q++) {
            jb->surf[p0 + q] = res[2 * q];
            jb->line[p0 + q] = res[2 * q + 1];
            jb->thick[p0 + q] = th[2 * q] | th[2 * q + 1];
        }
    }
    return NULL;
}

/* libm: 1 = stock gfortran -O3 (libmvec pairs for EXP of DO 20), 0 = scalar glibc exp.  Returns 0 on success. */
int spectrv_points(int64_t npts, int nrhox, int ifscat, int libm, const double *wave, const int64_t *iedge,
                   const double *wledge, const double *halfedge, const double *deledge, const double *contabs,
                   const double *contscat, const double *hkt, const double *rhox, const double *bfudge,
                   const double *fscat, const float *asynth, int64_t astride, const float *coefj, const float *ch,
                   const double *xtau8, double *surf, double *line, int *thick, int nthreads) {
    if (nrhox < 4 || nrhox > KWMAX) return -1;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    if (npts < 256 * nthreads) nthreads = (int)(npts / 256) > 0 ? (int)(npts / 256) : 1;
    job_t base = {nrhox, ifscat, libm, 0, wave, iedge, wledge, halfedge, deledge, contabs, contscat, hkt, rhox,
                  bfudge, fscat, asynth, astride, coefj, ch, xtau8, surf, line, thick, 0, npts};
    if (nthreads == 1) { run_job(&base); return 0; }
    job_t jobs[64];
    pthread_t th[64];
    int started[64];
    for (int t = 0; t < nthreads; t++) {
        jobs[t] = base;
        jobs[t].p0 = npts * t / nthreads;
        jobs[t].p1 = npts * (t + 1) / nthreads;
        started[t] = pthread_create(&th[t], NULL, run_job, &jobs[t]) == 0;
        if (!started[t]) run_job(&jobs[t]);
    }
    for (int t = 0; t < nthreads; t++) if (started[t]) pthread_join(th[t], NULL);
    return 0;
}
