/* DFSYNTHE MAIN line loop (dfsynthe.for labels 301-359) for a chunk of TAPE12 lines and all layers, REAL*4.

   A literal transcription of the Fortran loop (layer J outer, lines in file order inner, PROFILE computed until
   its first value below CONTMIN, then added to the red and blue wings), so the per-bin REAL*4 summation order and
   the first-crossing semantics are the Fortran's by construction.  The torch path (accumulate.py) is the
   reference implementation; this kernel only exists for speed (the cool temperatures have ~2.5e11 profile
   points).  fastmath != 0 selects the stock ifx -fp-model fast forms (see accumulate.py / dfsynthe.py):
     ADAMP>100 centre (K*C)/A;  1.4<ADAMP: PROFILEC=(K*.79788)*A, centre with R=1/AA, far wing with R=1/U;
     IV=NSTEP*(DVOIGT*200)+1.5;  X/NSTEP**2 via rcpps (table) + Newton step for the first (MAXSTEP-N10DOP)&~3 steps.
   Must be compiled without fast-math/FMA contraction (-O2 -ffp-contract=off); the FTZ/DAZ environment is the
   calling thread's (MXCSR), i.e. the caller's ftz.scope.  Build: python -m atlas_torch.dfsynthe.lines.native */
#include <stdint.h>
#include <string.h>
#include <math.h>

#define LENBUFF 3510000
#define LENGTH 3507859
#define MAXPROF 10000

static inline float rcpps_tab(float b, const uint32_t *tab) {
    uint32_t bits; memcpy(&bits, &b, 4);
    uint32_t m12 = (bits >> 11) & 0xFFF, e = (bits >> 23) & 0xFF, sign = bits & 0x80000000u;
    uint32_t tb = tab[m12];
    int te = (int)((tb >> 23) & 0xFF) - ((int)e - 127);
    uint32_t out;
    if (e == 0 || te >= 255) out = sign | 0x7F800000u;
    else if (te <= 0) out = sign;
    else out = (tb & 0x007FFFFFu) | ((uint32_t)te << 23) | sign;
    float r; memcpy(&r, &out, 4); return r;
}

static inline float div_nr(float a, float b, const uint32_t *tab) {
    float r = rcpps_tab(b, tab);
    float q0 = a * r;
    float t = q0 * b;
    float e = a - t;
    float s = e * r;
    return s + q0;
}

static inline float asym_src(float profc, float aa, float kdv) {
    float vv = kdv * kdv;
    float u = (aa + vv) * 1.4142f;
    float aau = aa / u, vvu = vv / u;
    float t = (aau - 10.0f * vvu) * aau + (5.0f * vvu) * vvu;
    t = t / u + vvu;
    t = t - aau / 3.0f;
    t = (t * 3.0f) / u + 1.0f;
    return (profc / u) * t;
}

static inline float asym_fast(float profc, float aa, float kdv) {
    float vv = kdv * kdv;
    float u = (vv + aa) * 1.4142f;
    float r = 1.0f / u;
    float aau = aa * r, vvu = vv * r;
    float t = (vvu * -10.0f + aau) * aau;
    t = t + (vvu * vvu) * 5.0f;
    float s = t * r + (aau * -0.33333334f + vvu);
    float p = (s * 3.0f) * r + 1.0f;
    return (r * profc) * p;
}

#define BLK 32
static float K2[MAXPROF + 1], RK2[MAXPROF + 1];
static const uint32_t *k2_tab = 0;

static void init_k2(const uint32_t *rcptab) {
    if (k2_tab == rcptab) return;
    for (int k = 1; k <= MAXPROF; k++) { float fk = (float)k; K2[k] = fk * fk; RK2[k] = rcpps_tab(K2[k], rcptab); }
    k2_tab = rcptab;
}

/* first k in [k0, MAXPROF] with profile[k] < contmin (else MAXPROF): profile values are computed in blocks of BLK
   (vectorisable, identical IEEE operations), the crossing is searched afterwards; values past it are unused. */
#define SCAN_BLOCKS(K0, EXPR)                                                        \
    do {                                                                             \
        nstep = MAXPROF;                                                             \
        for (int kb = (K0); kb <= MAXPROF; kb += BLK) {                              \
            int ke = kb + BLK - 1; if (ke > MAXPROF) ke = MAXPROF;                    \
            for (int k = kb; k <= ke; k++) profile[k] = (EXPR);                      \
            int hit = 0;                                                             \
            for (int k = kb; k <= ke; k++) if (profile[k] < contmin) { nstep = k; hit = 1; break; } \
            if (hit) break;                                                          \
        }                                                                            \
    } while (0)

/* AA state pre-pass: last ADAMP**2 of a line with ADAMP > 1.4 that passes the KAPPA0 test, per layer (in/out) */
int aa_scan(int nl, int n, const int32_t *nbuff, const float *congf, const int32_t *nelion,
            const float *gamr, const float *gams, const float *gamw,
            const float *xnfdop, const float *dopple, const float *xne, const float *txnxn,
            const float *cont, float *last)
{
    for (int j = 0; j < nl; j++) {
        const float *cj = cont + (size_t)j * LENGTH;
        const float *xdj = xnfdop + (size_t)j * 594, *dpj = dopple + (size_t)j * 594;
        for (int i = 0; i < n; i++) {
            const int nb = nbuff[i], nel = nelion[i];
            if (congf[i] * xdj[nel - 1] < cj[nb - 1]) continue;
            const float adamp = ((gamr[i] + gams[i] * xne[j]) + gamw[i] * txnxn[j]) / dpj[nel - 1];
            if (adamp > 1.4f) last[j] = adamp * adamp;
        }
    }
    return 0;
}

/* returns 0; stats[0] += items, stats[1] += profile points (1 + red + blue) */
int linesum(int nl, int n, const int32_t *nbuff, const float *congf, const int32_t *nelion,
            const float *gamr, const float *gams, const float *gamw,
            const float *xnfdop, const float *dopple, const float *xne, const float *txnxn,
            const float *cont, float *buf, float *aa_carry,
            const float *h0tab, const float *h1tab, const float *atab, const float *freq1000,
            const uint32_t *rcptab, int fastmath, float *profile, long long *stats)
{
    const double RESOLU = 500000.0;
    const float CB = 0.79788f / 1.4142f;                /* folded constant of the stock binary */
    long long items = 0, points = 0;
    init_k2(rcptab);
    for (int j = 0; j < nl; j++) {
        const float *cj = cont + (size_t)j * LENGTH;
        float *bj = buf + (size_t)j * LENBUFF;
        const float *xdj = xnfdop + (size_t)j * 594, *dpj = dopple + (size_t)j * 594;
        const float xnej = xne[j], txj = txnxn[j];
        float AA = aa_carry[j];
        for (int i = 0; i < n; i++) {
            const int nb = nbuff[i], nel = nelion[i];
            const float contmin = cj[nb - 1];
            const float kappa0 = congf[i] * xdj[nel - 1];
            if (kappa0 < contmin) continue;
            const float dop = dpj[nel - 1];
            const float adamp = ((gamr[i] + gams[i] * xnej) + gamw[i] * txj) / dop;
            int nstep;
            items++;
            if (adamp <= 0.2f) {                                            /* ---- A */
                bj[nb - 1] += kappa0 * (1.0f - 1.128f * adamp);
                const float tabstep = (float)((double)200.0f / ((double)dop * RESOLU));
                const int n10 = (int)((double)(10.0f * dop) * RESOLU);
                float tabi = 1.5f;
                int crossed = 0;
                nstep = n10;
                for (int k = 1; k <= n10; k++) {
                    tabi = tabi + tabstep;
                    int it = (int)tabi; if (it < 1) it = 1; if (it > 2001) it = 2001;
                    float p = kappa0 * (h0tab[it - 1] + adamp * h1tab[it - 1]);
                    profile[k] = p;
                    if (p < contmin) { nstep = k; crossed = 1; break; }
                }
                if (!crossed) {
                    float pn = n10 >= 1 ? profile[n10] : 0.0f;
                    float fn = (float)n10;
                    float X = pn * (fn * fn);
                    /* cvttss2si: out of range (or NaN) -> INT_MIN; MIN(MAXSTEP,MAXPROF) keeps it and the
                       wings are then skipped at label 323 (NBUFF+NSTEP < 1), as in the Fortran */
                    float v = sqrtf(X / contmin) + 1.0f;
                    int maxstep = (v < 2147483648.0f) ? (int)v : (int)0x80000000u;
                    if (maxstep > MAXPROF) maxstep = MAXPROF;
                    if (congf[i] * freq1000[nb / 1000] < 0.0001f || nel % 6 == 0)
                        if (maxstep > n10 * 10) maxstep = n10 * 10;
                    int vend = n10;
                    if (fastmath && maxstep > n10) vend = n10 + ((maxstep - n10) & ~3);
                    int k = n10 + 1;
                    for (; k <= vend && k <= maxstep; k++) {          /* rcpps + Newton step (stock vector loop) */
                        float q0 = X * RK2[k];
                        profile[k] = (X - q0 * K2[k]) * RK2[k] + q0;
                    }
                    for (; k <= maxstep; k++) profile[k] = X / K2[k];
                    nstep = maxstep;
                }
            } else if (adamp > 100.0f) {                                    /* ---- B */
                const float pc = fastmath ? (kappa0 * CB) / adamp : ((kappa0 / adamp) * 0.79788f) / 1.4142f;
                bj[nb - 1] += pc;
                const float aa = adamp * adamp;
                AA = aa;
                const float dv = (float)((double)(1.0f / dop) / RESOLU);
                const float pcaa = pc * aa;
                SCAN_BLOCKS(1, pcaa / (aa + ((float)k * dv) * ((float)k * dv)));
            } else if (adamp > 1.4f) {                                      /* ---- C */
                const float aa = adamp * adamp;
                AA = aa;
                float profc, cen;
                if (fastmath) {
                    profc = (kappa0 * 0.79788f) * adamp;
                    float r = 1.0f / aa;
                    cen = ((((r * 0.75f) + -0.5f) * r + 1.0f) * (kappa0 * CB)) / adamp;
                } else {
                    profc = (kappa0 * adamp) * 0.79788f;
                    cen = (((kappa0 / adamp) * 0.79788f) / 1.4142f) * (1.0f + ((0.75f / aa) - 0.5f) / aa);
                }
                bj[nb - 1] += cen;
                const float dv = (float)((double)(1.0f / dop) / RESOLU);
                if (fastmath) SCAN_BLOCKS(1, asym_fast(profc, aa, (float)k * dv));
                else SCAN_BLOCKS(1, asym_src(profc, aa, (float)k * dv));
            } else {                                                        /* ---- D */
                int ia = (int)(adamp * 200.0f + 1.5f);
                const float *arow = atab + (size_t)(ia - 1) * 1001;
                bj[nb - 1] += kappa0 * arow[0];
                const float dv = (float)((double)(1.0f / dop) / RESOLU);
                const int n5 = (int)((double)(5.0f * dop) * RESOLU);
                int crossed = 0;
                const float dv200 = dv * 200.0f;
                for (int k = 1; k <= n5; k++) {
                    int iv = fastmath ? (int)((float)k * dv200 + 1.5f) : (int)(((float)k * dv) * 200.0f + 1.5f);
                    if (iv < 1) iv = 1;
                    if (iv > 1001) iv = 1001;
                    float p = kappa0 * arow[iv - 1];
                    profile[k] = p;
                    if (p < contmin) { nstep = k; crossed = 1; break; }
                }
                if (!crossed) {
                    const float profc = fastmath ? (kappa0 * 0.79788f) * adamp : (kappa0 * adamp) * 0.79788f;
                    const float aas = AA;
                    if (fastmath) SCAN_BLOCKS(n5 + 1, asym_fast(profc, aas, (float)k * dv));
                    else SCAN_BLOCKS(n5 + 1, asym_src(profc, aas, (float)k * dv));
                }
            }
            /* label 323: red and blue wings */
            int r = (nb < LENGTH) ? ((LENGTH - nb) < nstep ? (LENGTH - nb) : nstep) : 0;
            int b = (nb > 1) ? ((nb - 1) < nstep ? (nb - 1) : nstep) : 0;
            if (r < 0) r = 0;
            if (b < 0) b = 0;
            float *c0 = bj + (nb - 1);
            for (int k = 1; k <= r; k++) c0[k] += profile[k];
            for (int k = 1; k <= b; k++) c0[-k] += profile[k];
            points += 1 + r + b;
        }
        aa_carry[j] = AA;
    }
    stats[0] += items;
    stats[1] += points;
    return 0;
}

/* ------------------------------------------------------------------------------------------------------------
   XLINOP normal lines (label 200: TYPE 0, 3 and < -2) for a run of consecutive TAPE19 lines and all layers:
   VOIGT profile, WAVE chains (REAL*8 WAVE=WAVE*RATIO / WAVE/RATIO), merged-continuum tails (WCON, WTAIL),
   first-crossing termination after the add, red then blue wing, layer J outer, lines in order.  Transcribes
   xlinop.py (XLinop.normal_lines/_walk/voigt), which is validated against the Fortran oracle. */
static inline float voigt_f(float v, float a, const float *h0, const float *h1, const float *h2, const float *atab) {
    if (a < 0.2f) {
        if (v > 10.0f) return (0.5642f * a) / (v * v);
        int iv = (int)(v * 200.0f + 1.5f);
        if (iv < 1) iv = 1;
        if (iv > 2001) iv = 2001;
        return (h2[iv - 1] * a + h1[iv - 1]) * a + h0[iv - 1];
    }
    if (a > 1.4f || (a + v) > 3.2f) {
        float aa = a * a, vv = v * v;
        float u = (aa + vv) * 1.4142f;
        float r0 = (a * 0.79788f) / u;
        if (a > 100.0f) return r0;
        float aau = aa / u, vvu = vv / u;
        float t = (aau - 10.0f * vvu) * aau + (5.0f * vvu) * vvu;
        t = t / u + vvu;
        t = t - aau / 3.0f;
        return r0 * ((t * 3.0f) / u + 1.0f);
    }
    int ia = (int)(a * 200.0f + 1.5f), iv = (int)(v * 200.0f + 1.5f);
    if (ia < 1) ia = 1;
    if (ia > 281) ia = 281;
    if (iv < 1) iv = 1;
    if (iv > 1001) iv = 1001;
    return atab[(size_t)(ia - 1) * 1001 + (iv - 1)];
}

int xlinop_normal(int nl, int n, const double *wl, const float *gf, const int32_t *nbuff, const int32_t *nelion,
                  const int32_t *ncon, const int32_t *nelionx, const float *gammar, const float *gammas,
                  const float *gammaw, const double *w0red, const double *w0blue,
                  const float *xnfdop, const float *dopple, const float *xne, const float *txnxn,
                  const double *emerge, const double *contx, const float *cont, float *buf,
                  const float *h0, const float *h1, const float *h2, const float *atab,
                  double ratio, double wlbeg, double wlend, long long *stats)
{
    long long items = 0, contribs = 0;
    for (int j = 0; j < nl; j++) {
        const float *cj = cont + (size_t)j * LENGTH;
        float *bj = buf + (size_t)j * LENBUFF;
        const float *xdj = xnfdop + (size_t)j * 594, *dpj = dopple + (size_t)j * 594;
        for (int i = 0; i < n; i++) {
            const int nb = nbuff[i], nel = nelion[i];
            const float kappa0 = gf[i] * xdj[nel - 1];
            const float cnb = cj[nb - 1];
            if (kappa0 < cnb) continue;
            items++;
            const int strong = kappa0 > 500.0f * cnb;
            double wc = 0.0, wt = 0.0;
            if (ncon[i] > 0) {
                int nc = ncon[i], nx = nelionx[i];
                if (nc > 26) nc = 26;
                if (nx < 1) nx = 1;
                if (nx > 17) nx = 17;
                double cx = contx[(size_t)(nc - 1) * 17 + (nx - 1)];
                wc = 1.0e7 / (cx - emerge[j]);
                wt = 1.0e7 / ((cx - emerge[j]) - 500.0);
            }
            const float dop = dpj[nel - 1];
            const float a = ((gammar[i] + gammas[i] * xne[j]) + gammaw[i] * txnxn[j]) / dop;
            const float dopwl = (float)((double)dop * wl[i]);
            const double wli = wl[i];
            const int red_ok = !(wli > wlend);
            int minred = nb > 1 ? nb : 1;
            if (red_ok) {
                int maxred = strong ? LENGTH : (nb + 1000 < LENGTH ? nb + 1000 : LENGTH);
                double wave = w0red[i];
                for (int ib = minred; ib <= maxred; ib++) {
                    if (wave < wc) { wave = wave * ratio; continue; }
                    float vv = (float)(fabs(wave - wli) / (double)dopwl);
                    float kv = kappa0 * voigt_f(vv, a, h0, h1, h2, atab);
                    if (wave < wt) kv = (float)(((double)kv * (wave - wc)) / (wt - wc));
                    bj[ib - 1] += kv;
                    contribs++;
                    if (kv < cj[ib - 1]) break;
                    wave = wave * ratio;
                }
            }
            if (red_ok && (minred == 1 || wli < wlbeg)) continue;
            {
                int ib = nb < LENGTH + 1 ? nb : LENGTH + 1;
                int maxblue = ib - 1;
                if (!strong && maxblue > 1000) maxblue = 1000;
                double wave = w0blue[i];
                for (int k = 1; k <= maxblue; k++) {
                    ib--;
                    wave = wave / ratio;
                    if (wave < wc) break;            /* WAVE decreases: every later bin is skipped too */
                    float vv = (float)(fabs(wave - wli) / (double)dopwl);
                    float kv = kappa0 * voigt_f(vv, a, h0, h1, h2, atab);
                    if (wave < wt) kv = (float)(((double)kv * (wave - wc)) / (wt - wc));
                    bj[ib - 1] += kv;
                    contribs++;
                    if (kv < cj[ib - 1]) break;
                }
            }
        }
    }
    stats[0] += items;
    stats[1] += contribs;
    return 0;
}
