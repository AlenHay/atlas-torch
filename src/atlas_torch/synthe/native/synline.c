/* SYNTHE (synthe.for, BasicATLAS patch) line loops in C, bit-identical to the torch port they accelerate
   (atlas_torch.synthe.lineopac.xlinop / synthe_prog).  REAL*4 where the Fortran is REAL*4, same operation order,
   no FMA contraction, no fast-math (build: python -m atlas_torch.synthe.native, or `make native`).

   xl_normal: XLINOP labels 200-214 (TYPE 0/3 "normal" line with Voigt wings) for one fort.19 line and the layers
   of a block that passed the KAPMIN tests: red wing from MINRED (stop before adding the first point below
   CONTINUUM*CUTOFF), blue wing from MIN(LENGTH+1,NBUFF)-1 down (add, then stop), WAVE < WCON points skipped,
   WCON..WTAIL taper.  Lines are called in fort.19 order, so every BUFFER element receives its terms in the
   Fortran's order. */
#include <math.h>
#include <pthread.h>
#include <stdint.h>

/* run fn(arg, t, nt) on nt threads (t = 0..nt-1; thread 0 is the caller) */
typedef void (*part_fn)(void *arg, int t, int nt);
typedef struct { part_fn fn; void *arg; int t, nt; } part_t;
static void *part_run(void *p) { part_t *q = (part_t *)p; q->fn(q->arg, q->t, q->nt); return NULL; }
static void run_parts(part_fn fn, void *arg, int nt) {
    if (nt > 64) nt = 64;
    if (nt <= 1) { fn(arg, 0, 1); return; }
    pthread_t th[64]; part_t pa[64]; int ok[64];
    for (int t = 1; t < nt; t++) {
        pa[t] = (part_t){fn, arg, t, nt};
        ok[t] = pthread_create(&th[t], NULL, part_run, &pa[t]) == 0;
        if (!ok[t]) fn(arg, t, nt);
    }
    fn(arg, 0, nt);
    for (int t = 1; t < nt; t++) if (ok[t]) pthread_join(th[t], NULL);
}

/* FUNCTION VOIGT(V,A), COMMON /H1TAB/ H0TAB(2001),H1TAB(2001),H2TAB(2001) = h[0..6002] */
static inline float voigt(float v, float a, const float *h) {
    const float *h0 = h - 1, *h1 = h + 2000, *h2 = h + 4001;     /* 1-based */
    if (a < .2f) {
        if (v > 10.f) return .5642f * a / (v * v);
        int iv = (int)(v * 200.f + 1.5f);
        return (h2[iv] * a + h1[iv]) * a + h0[iv];
    }
    if (a > 1.4f || a + v > 3.2f) {
        float aa = a * a, vv = v * v;
        float u = (aa + vv) * 1.4142f;
        float vo = a * .79788f / u;
        if (a > 100.f) return vo;
        float aau = aa / u, vvu = vv / u, uu = u * u;
        return ((((aau - 10.f * vvu) * aau * 3.f + 15.f * vvu * vvu) + 3.f * vv - aa) / uu + 1.f) * vo;
    }
    int iv = (int)(v * 200.f + 1.5f);
    float vv = v * v;
    float hh1 = h1[iv] + h0[iv] * 1.12838f;
    float hh2 = h2[iv] + hh1 * 1.12838f - h0[iv];
    float hh3 = (1.f - h2[iv]) * .37613f - hh1 * .66667f * vv + hh2 * 1.12838f;
    float hh4 = (3.f * hh3 - hh1) * .37613f + h0[iv] * .66667f * vv * vv;
    return ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0[iv]) *
           (((-.122727278f * a + .532770573f) * a - .96284325f) * a + .979895032f);
}

/* libgcc __powidf2(x, m), m >= 0 (gfortran REAL*8**INTEGER) */
static inline double powi(double x, int64_t m) {
    int64_t n = m;
    double y = (n % 2) ? x : 1.;
    while (n >>= 1) {
        x = x * x;
        if (n % 2) y = y * x;
    }
    return y;
}

typedef struct {
    int nl; const int64_t *lay; const float *kappa0, *adamp, *dopwl; const double *wcon, *wtail;
    float wl4; double wl; int64_t nbuff, length; double wbegin, ratio, wlend; float *buffer;
    const float *cutcont, *h;
} xln_t;

static void xl_normal_part(void *arg, int t, int nt) {
    const xln_t *X = (const xln_t *)arg;
    const int64_t *lay = X->lay; const float *kappa0 = X->kappa0, *adamp = X->adamp, *dopwl = X->dopwl;
    const double *wcon = X->wcon, *wtail = X->wtail; const float wl4 = X->wl4; const double wl = X->wl;
    const int64_t nbuff = X->nbuff, length = X->length; const double wbegin = X->wbegin, ratio = X->ratio;
    const double wlend = X->wlend; float *buffer = X->buffer; const float *cutcont = X->cutcont, *h = X->h;
    for (int i = t; i < X->nl; i += nt) {
        float *buf = buffer + lay[i] * length - 1;                 /* 1-based BUFFER(IBUFF) of this layer */
        const float *cc = cutcont + lay[i] * length - 1;
        const float k0 = kappa0[i], a = adamp[i], dw = dopwl[i];
        const double wc = wcon[i], wt = wtail[i];
        if (!(wl > wlend)) {                                       /* RED WING */
            int64_t minred = nbuff > 1 ? nbuff : 1;
            double wave = wbegin * powi(ratio, minred - 1);
            float wave4 = (float)wave;
            for (int64_t ib = minred; ib <= length; ib++) {
                if (!(wave < wc)) {
                    float vv = fabsf(wave4 - wl4) / dw;
                    float kappa = k0 * voigt(vv, a, h);
                    if (wave < wt) kappa = (float)((double)kappa * (wave - wc) / (wt - wc));
                    if (kappa < cc[ib]) break;
                    buf[ib] = buf[ib] + kappa;
                }
                wave = wave * ratio;
                wave4 = (float)wave;
            }
            if (minred == 1 || wl < wbegin) continue;
        }
        int64_t ib = nbuff < length + 1 ? nbuff : length + 1;      /* BLUE WING (label 213) */
        int64_t maxblue = ib - 1;
        if (maxblue < 1) continue;
        double wave = wbegin * powi(ratio, ib - 1);
        for (int64_t k = 1; k <= maxblue; k++) {
            ib = ib - 1;
            wave = wave / ratio;
            float wave4 = (float)wave;
            if (wave < wc) continue;
            float vv = fabsf(wave4 - wl4) / dw;
            float kappa = k0 * voigt(vv, a, h);
            if (wave < wt) kappa = (float)((double)kappa * (wave - wc) / (wt - wc));
            buf[ib] = buf[ib] + kappa;
            if (kappa < cc[ib]) break;
        }
    }
}

/* layers are independent (each touches only its BUFFER row): nthreads > 1 splits them over threads */
int xl_normal(int nl, const int64_t *lay, const float *kappa0, const float *adamp, const float *dopwl,
              const double *wcon, const double *wtail, float wl4, double wl, int64_t nbuff, int64_t length,
              double wbegin, double ratio, double wlend, float *buffer, const float *cutcont, const float *h,
              int nthreads) {
    xln_t X = {nl, lay, kappa0, adamp, dopwl, wcon, wtail, wl4, wl, nbuff, length, wbegin, ratio, wlend, buffer,
               cutcont, h};
    int nt = nthreads < nl / 2 ? nthreads : nl / 2;
    run_parts(xl_normal_part, &X, nt < 1 ? 1 : nt);
    return 0;
}

/* MAIN line loop, buffer update (labels 321-326) for a batch of (layer, line) pairs in the Fortran's order (layer
   major, fort.12 order within a layer): pair p covers the points NBUFF-NSTEP..NBUFF+NSTEP clipped to 1..LENGTH and
   adds KAPCEN at the centre, PROFILE(|offset|) in the wings: the core values for |offset| <= N10DOP, X/offset**2
   (REAL*4) beyond.  Each pair adds at most once per point, so pair order = per-point Fortran order. */
typedef struct {
    int64_t npair; const int64_t *jj, *nbs, *nstep, *n10; const float *kapcen, *x, *core; int64_t core_stride, length;
    float *buffer; int64_t nlay;
} macc_t;

static void macc_part(void *arg, int t, int nt) {
    const macc_t *M = (const macc_t *)arg;
    const int64_t *jj = M->jj, *nbs = M->nbs, *nstep = M->nstep, *n10 = M->n10;
    const float *kapcen = M->kapcen, *x = M->x, *core = M->core;
    const int64_t core_stride = M->core_stride, length = M->length;
    float *buffer = M->buffer;
    for (int64_t p = 0; p < M->npair; p++) {
        if (nt > 1 && jj[p] % nt != t) continue;          /* thread t owns layers j = t mod nt */
        int64_t nb = nbs[p], ns = nstep[p], nd = n10[p];
        int64_t lo = nb - ns > 1 ? nb - ns : 1;
        int64_t hi = nb + ns < length ? nb + ns : length;
        if (hi < lo) continue;
        float *buf = buffer + jj[p] * length - 1;                    /* 1-based */
        const float *cr = core + p * core_stride - 1;                /* PROFILE(1..) of the core */
        const float xp = x[p], kc = kapcen[p];
        for (int64_t pnt = lo; pnt <= hi; pnt++) {
            int64_t off = pnt > nb ? pnt - nb : nb - pnt;
            float v;
            if (off == 0) v = kc;
            else if (off <= nd) v = cr[off];
            else { float o = (float)off; v = xp / (o * o); }
            buf[pnt] = buf[pnt] + v;
        }
    }
}

/* pairs of different layers touch different BUFFER rows: nthreads > 1 splits the layers over threads, each thread
   keeping the pair order within its layers */
int main_accumulate(int64_t npair, const int64_t *jj, const int64_t *nbs, const int64_t *nstep, const int64_t *n10,
                    const float *kapcen, const float *x, const float *core, int64_t core_stride, int64_t length,
                    float *buffer, int nthreads) {
    macc_t M = {npair, jj, nbs, nstep, n10, kapcen, x, core, core_stride, length, buffer, 0};
    run_parts(macc_part, &M, npair < 100000 ? 1 : nthreads);
    return 0;
}

/* ============================================================================================================
   XLINOP hydrogen lines (synthe.for labels 600-634, HPROF4 / SOFBET / VCSE1F / FASTEX of synthe.for) for one
   fort.19 line and the layers of a block that passed the KAPMIN tests: every point of both wings (no sampling).
   Transcribes lineopac/xlinop.py XLinop._hydrogen and lineopac/hydrogen.py HProf4 operation by operation (REAL*4,
   the REAL*8 sub-expressions where the port has them); LOGF/POWF are glibc's scalar functions (what libm32.slogf
   / spowf call), so the kernel is bit-identical to the torch path by construction. */
#include <stdlib.h>

typedef struct {                                   /* HProf4.line(n, m) constants + per-layer Stark set-up */
    int n, m, nfin;
    float freqnm, dbeta, stark, vdw, vdw2, radamp, resont, radfq_pi, fqrad;
    double wavenm;
    float finest[16], finswt[16];
    const float *c1, *c2, *g1, *gnot;              /* [block layers] */
} shs_t;

typedef struct {
    const float *fo, *t3nhe, *t3nh2, *xnfph1, *xnfph2, *gcon1, *gcon2, *pp;    /* [block layers] */
    const float *dopph;                            /* DOPPH(J) state [block layers] */
    const float *extab; int64_t nextab;
    const float *cutoffh2, *cutoffh2plus;          /* 91, 111 */
    const float *prob, *sc, *sd, *spp, *sbeta;     /* 525, 35, 35, 5, 15 */
    float c4000;                                   /* 10.**(-11.07-14.) folded by gfortran */
} shd_t;

typedef struct {
    int nl, path, do_red;                          /* path 0 alpha, 1 beta, 2 general */
    const int64_t *lay;                            /* block-relative layer of item i */
    const float *k0, *k0red, *k0blue;              /* per item */
    const double *wcon, *wtail;                    /* per item (general) */
    int64_t nbuff, length;
    double wl, wbegin, ratio, alphahyd, redcut, wlm1, wlm2, bluecut, wlp1, wlp2;
    const shs_t *s_main, *s_red, *s_blue;
    float *buffer; const float *cutcont;           /* [block layers, LENGTH] */
} shl_t;

static inline int64_t f2i(float x) { return (x > -9.2e18f && x < 9.2e18f) ? (int64_t)x : INT64_MIN; }
static inline int64_t clampi(int64_t i, int64_t n) { return i < 0 ? 0 : (i > n - 1 ? n - 1 : i); }

static inline float fastex_s(float x, const float *ext, int64_t n) {
    float ix = truncf(x);
    int64_t i1 = f2i(ix);
    float fr = (x - ix) * 1000.f;
    fr = fr + 1.5f;
    int64_t i2 = f2i(truncf(fr));
    int64_t k2 = i2 == INT64_MIN ? 0 : 1000 + i2;
    return ext[clampi(i1 == INT64_MIN ? 0 : i1, n)] * ext[clampi(k2, n)];
}

static float vcse1f_s(float x, const float *ext, int64_t n) {
    if (x > 0.f && x <= 1.f) {
        float lg = logf(x);
        if (x <= 0.01f) return (-lg - .577215f) + x;
        float p = x * .00107857f;
        p = -.00976004f + p; p = x * p;
        p = .05519968f + p; p = x * p;
        p = -.24991055f + p; p = x * p;
        p = .99999193f + p;
        return (-lg - .57721566f) + x * p;
    }
    if (x > 1.f && x <= 30.f)
        return (((x * (x + 2.334733f) + .25062f) / (x * (x + 3.330657f) + 1.681534f)) / x) * fastex_s(x, ext, n);
    return 0.f;
}

static float sofbet_s(const shd_t *H, float b, float p, int n, int m) {
    float b2 = b * b, sb = sqrtf(b);
    int indx = 7, mmn = m - n;
    if (n <= 3 && mmn <= 2) indx = 2 * (n - 1) + mmn;
    int64_t im = f2i(truncf(5.f * p));
    im = (im == INT64_MIN ? im : im + 1);
    if (im > 4) im = 4;
    int64_t ip = im + 1;
    float wtpp = 5.f * (p - H->spp[clampi(im - 1, 5)]);
    float wtpm = 1.f - wtpp;
    float asym = (1.5f / sb + 27.f / b2) / b2;
    if (b > 500.f) return asym * 1.f;
    int ci = (indx - 1) * 5;
    if (b > 25.12f) {
        float cc = H->sc[clampi(ci + ip - 1, 35)] * wtpp + H->sc[clampi(ci + im - 1, 35)] * wtpm;
        float dd = H->sd[clampi(ci + ip - 1, 35)] * wtpp + H->sd[clampi(ci + im - 1, 35)] * wtpm;
        return asym * (1.f + dd / (cc + b * sb));
    }
    float bc = b < H->sbeta[14] ? b : H->sbeta[14];
    int jj = 15;
    for (int J = 2; J <= 15; J++) if (bc <= H->sbeta[J - 1]) { jj = J; break; }
    int jm = jj - 1, jp = jj;
    float wtbp = (b - H->sbeta[jm - 1]) / (H->sbeta[jp - 1] - H->sbeta[jm - 1]);
    float wtbm = 1.f - wtbp;
    int k0 = 75 * (indx - 1);
#define PR(i, j) H->prob[clampi((i) - 1 + 5 * ((j) - 1) + k0, 525)]
    float cbp = PR(ip, jp) * wtpp + PR(im, jp) * wtpm;
    float cbm = PR(ip, jm) * wtpp + PR(im, jm) * wtpm;
#undef PR
    float corr = (1.f + cbp * wtbp) + cbm * wtbm;
    float wt = .5f * (10.f - b);
    wt = wt < 1.f ? wt : 1.f;
    wt = wt < 0.f ? 0.f : wt;
    float pr1 = b <= 10.f ? 8.f / (83.f + (2.f + .95f * b2) * b) : 0.f;
    float pr2 = b >= 8.f ? asym : 0.f;
    return (pr1 * wt + pr2 * (1.f - wt)) * corr;
}

/* HPROF4(N,M,J,DELW,DOPPH) of synthe.for, layer j (block-relative) */
static float hprof4_s(const shd_t *H, const shs_t *c, int64_t j, double delw) {
    const int n = c->n, m = c->m;
    const float fq = c->freqnm;
    const double wn = c->wavenm;
    float delstark = (float)(((-10.0 * delw) / wn) * (double)fq);
    float wl = (float)(wn + delw * 10.0);
    float freq = (float)(2.99792458e18 / (double)wl);
    float dl = fabsf(freq - fq);
    float fo = H->fo[j];
    float hwstk = c->stark * fo;
    float hwvdw = c->vdw * H->t3nhe[j] + c->vdw2 * H->t3nh2[j];
    float hwrad = c->radamp;
    float hwres = (c->resont * H->xnfph1[j]) * 2.f;
    float hwlor = (hwres + hwvdw) + hwrad;
    float hwdop = H->dopph[j];
    int nwid = (hwdop >= hwstk && hwdop >= hwlor) ? 1 : (hwlor >= hwstk ? 2 : 3);
    float mx = (hwdop != hwdop || hwlor != hwlor) ? NAN : (hwdop > hwlor ? hwdop : hwlor);
    mx = (mx != mx || hwstk != hwstk) ? NAN : (mx > hwstk ? mx : hwstk);
    float hfwid = fq * mx;
    int ifcore = dl <= hfwid;
    float dop = fq * hwdop;
    float hd = 0.f, hlor = 0.f, extra = 0.f, mainv = 0.f;
    int need_d = !ifcore || nwid == 1, need_l = !ifcore || nwid == 2, need_s = !ifcore || nwid == 3;
    if (need_d) {
        float df = freq - fq;
        for (int i = 0; i < c->nfin; i++) {
            float dd = fabsf(df - c->finest[i]) / dop;
            if (dd <= 7.f) hd = hd + fastex_s(dd * dd, H->extab, H->nextab) * c->finswt[i];
        }
    }
    if (need_l) {
        double fd = (double)freq;
        if (n == 1 && m == 2) {
            float hwres4 = hwres * 4.f;
            float hwlor4 = (hwres4 + hwvdw) + hwrad;
            float hhw = fq * hwlor4;
            float den = dl * dl + hhw * hhw;
            float hres;
            if (fd > (82259.105 - 4000.0) * 2.99792458e10) {
                hres = (((hwres4 * fq) / 3.14159f) / den * 1.77245f) * dop;
            } else {
                const float spacing = (float)(200. * 2.99792458e10), f22 = (float)((82259.105 - 22000.0) * 2.99792458e10);
                const float *ch2 = H->cutoffh2;
                float ff = freq, cut;
                if (ff < f22) cut = (ch2[1] - ch2[0]) / spacing * (ff - f22) + ch2[0];
                else {
                    int64_t icut = f2i(truncf((ff - f22) / spacing));
                    icut = icut < 0 ? 0 : (icut > 89 ? 89 : icut);
                    float cutfreq = (float)icut * spacing + f22;
                    cut = (ch2[icut + 1] - ch2[icut]) / spacing * (ff - cutfreq) + ch2[icut];
                }
                float p = powf(10.f, cut - 14.f);
                cut = (float)((double)((p * H->xnfph1[j]) * 2.f) / 2.99792458e10);
                if (fd < 50000.0 * 2.99792458e10) cut = 0.f;
                hres = (cut * 1.77245f) * dop;
            }
            float hrad = 0.f;
            if (fd > 2.4190611e15 && fd < (double).77f * 3.288051e15)
                hrad = (c->radfq_pi / den * 1.77245f) * dop;
            float hvdw = (((hwvdw * fq) / 3.14159f) / den * 1.77245f) * dop;
            if (fd < 1.8e15) hvdw = 0.f;
            hlor = (hres + hrad) + hvdw;
        } else {
            float hhw = fq * hwlor;
            float top = hhw;
            if (n == 1 && m >= 3 && m <= 5) {
                float lo = m == 3 ? .885f : (m == 4 ? .936f : .959f);
                float hi = m == 3 ? .890f : (m == 4 ? .938f : .961f);
                if (fd > (double)lo * 3.288051e15 && fd < (double)hi * 3.288051e15) top = hhw - c->fqrad;
            }
            hlor = ((top / 3.14159f) / (dl * dl + hhw * hhw) * 1.77245f) * dop;
        }
    }
    if (need_s) {
        float c1 = c->c1[j], c2 = c->c2[j], g1 = c->g1[j], gnot = c->gnot[j];
        float gcon1 = H->gcon1[j], gcon2 = H->gcon2[j];
        float dbeta = c->dbeta;
        float beta = (fabsf(delstark) / fo) * dbeta;
        float y1 = c1 * beta;
        float y2 = c2 * (beta * beta);
        int skip = (y2 <= 1e-4f) && (y1 <= 1e-5f);
        float t90 = 90.f * y1;
        float y1c = y1 > 80.f ? 80.f : y1;
        float gam = g1 * ((0.5f * fastex_s(y1c, H->extab, H->nextab) + vcse1f_s(y1, H->extab, H->nextab)) -
                          0.5f * vcse1f_s(y2, H->extab, H->nextab)) *
                    ((1.f - gcon1 / (1.f + t90 * t90 * t90)) - gcon2 / (1.f + 2000.f * y1));
        if (gam <= 1e-20f) gam = 0.f;
        if (skip) gam = gnot;
        float prqs = sofbet_s(H, beta, H->pp[j], n, m);
        if (m <= 2) {
            prqs = prqs * 0.5f;
            double fd = (double)freq;
            int lo = fd < ((double)82259.105f - 20000.0) * 2.99792458e10;
            int hi = fd > ((double)82259.105f - 4000.0) * 2.99792458e10;
            if (lo) extra = 0.f;
            else if (hi) {
                float beta4000 = (float)((4000.0 * 2.99792458e10 / (double)fo) * (double)dbeta);
                float prqsp4000 = ((sofbet_s(H, beta4000, H->pp[j], n, m) * 0.5f) / fo) * dbeta;
                float cut4000 = (float)(((double)H->c4000 / 2.99792458e10) * (double)H->xnfph2[j]);
                extra = (((((cut4000 / prqsp4000) * prqs) / fo) * dbeta) * 1.77245f) * dop;
            } else {
                const float f15 = (float)(((double)82259.105f - 15000.0) * 2.99792458e10);
                const float spacing = (float)(100. * 2.99792458e10);
                const float *ch = H->cutoffh2plus;
                float ff = freq, cut;
                if (ff < f15) cut = (ch[1] - ch[0]) / spacing * (ff - f15) + ch[0];
                else {
                    int64_t icut = f2i(truncf((ff - f15) / spacing));
                    icut = icut < 0 ? 0 : (icut > 109 ? 109 : icut);
                    float cutfreq = (float)icut * spacing + f15;
                    cut = (ch[icut + 1] - ch[icut]) / spacing * (ff - cutfreq) + ch[icut];
                }
                float p = powf(10.f, cut - 14.f);
                float cutv = (float)(((double)p / 2.99792458e10) * (double)H->xnfph2[j]);
                extra = (cutv * 1.77245f) * dop;
            }
        }
        float ffv = gam > 0.f ? (gam / 3.14159f) / (gam * gam + beta * beta) : 0.f;
        float p1 = .9f * y1;
        p1 = p1 * p1;
        float fns = (p1 + .03f * sqrtf(y1)) / (p1 + 1.f);
        mainv = (((((prqs * (1.f + fns) + ffv) / fo) * dbeta) * 1.77245f) * dop);
    }
    if (ifcore) {
        if (nwid == 1) return hd;
        if (nwid == 2) return 0.f + hlor;
        return (0.f + extra) + mainv;
    }
    return ((hd + hlor) + extra) + mainv;
}

static inline float tailf(float k, double w, double wc, double wt) {
    return w < wt ? (float)(((double)k * (w - wc)) / (wt - wc)) : k;
}

/* red wing from bin `start` (WAVE(start) = w0): returns 1 if the walk stopped at ALPHAHYD (beta path) */
static int h_red(const shd_t *H, const shl_t *L, int i, int64_t start) {
    const int64_t j = L->lay[i], length = L->length;
    float *buf = L->buffer + j * length - 1;
    const float *cc = L->cutcont + j * length - 1;
    const float k0 = L->k0[i];
    const int general = L->path == 2, beta = L->path == 1;
    const double wc = general ? L->wcon[i] : 0., wt = general ? L->wtail[i] : 0.;
    double wave = L->wbegin * powi(L->ratio, start - 1);
    for (int64_t ib = start; ib <= length; ib++) {
        if (ib > start) wave = wave * L->ratio;
        if (general) {
            if (wave < wc) continue;
            if (wave > L->wlm1) return 0;
        }
        if (beta && wave > L->alphahyd) return 1;
        float k = k0 * hprof4_s(H, L->s_main, j, wave - L->wl);
        if (general) {
            k = tailf(k, wave, wc, wt);
            if (wave > L->redcut) {
                float kr = tailf(L->k0red[i] * hprof4_s(H, L->s_red, j, wave - L->wlm2), wave, wc, wt);
                if (kr >= k) return 0;
            }
        }
        if (k < cc[ib]) return 0;
        buf[ib] = buf[ib] + k;
    }
    return 0;
}

/* blue wing from IBUFF = ib0 (first point ib0-1) */
static void h_blue(const shd_t *H, const shl_t *L, int i, int64_t ib0) {
    if (ib0 - 1 < 1) return;
    const int64_t j = L->lay[i], length = L->length;
    float *buf = L->buffer + j * length - 1;
    const float *cc = L->cutcont + j * length - 1;
    const float k0 = L->k0[i];
    const int general = L->path == 2;
    const double wc = general ? L->wcon[i] : 0., wt = general ? L->wtail[i] : 0.;
    double wave = L->wbegin * powi(L->ratio, ib0 - 1);
    for (int64_t ib = ib0 - 1; ib >= 1; ib--) {
        wave = wave / L->ratio;
        if (general && (wave < wc || wave < L->wlp1)) return;
        float k = k0 * hprof4_s(H, L->s_main, j, wave - L->wl);
        if (general) {
            k = tailf(k, wave, wc, wt);
            if (wave < L->bluecut) {
                float kb = tailf(L->k0blue[i] * hprof4_s(H, L->s_blue, j, wave - L->wlp2), wave, wc, wt);
                if (kb > k) return;
            }
        }
        buf[ib] = buf[ib] + k;
        if (k < cc[ib]) return;
    }
}

typedef struct { const shd_t *H; const shl_t *L; } shjob_t;

static void xl_hydrogen_part(void *arg, int t, int nt) {
    const shjob_t *J = (const shjob_t *)arg;
    const shd_t *H = J->H; const shl_t *L = J->L;
    const int64_t nb = L->nbuff, length = L->length;
    const int64_t minred = nb > 1 ? nb : 1, ib0 = nb < length + 1 ? nb : length + 1;
    for (int i = t; i < L->nl; i += nt) {
        if (L->path == 1) {                                  /* beta lines (630-634) */
            if (!L->do_red) { h_blue(H, L, i, ib0); continue; }
            int reached = minred <= length ? h_red(H, L, i, minred) : 0;
            if (reached) { if (nb >= 1) h_blue(H, L, i, nb); continue; }
            if (minred == 1 || L->wl < L->wbegin) continue;    /* 622 */
            h_blue(H, L, i, ib0);
            continue;
        }
        if (L->do_red) {
            if (minred <= length) h_red(H, L, i, minred);
            if (minred == 1 || L->wl < L->wbegin) continue;
        }
        h_blue(H, L, i, ib0);
    }
}

int xl_hydrogen(const shd_t *H, const shl_t *L, int nthreads) {
    shjob_t J = {H, L};
    int nt = nthreads < L->nl / 2 ? nthreads : L->nl / 2;
    run_parts(xl_hydrogen_part, &J, nt < 1 ? 1 : nt);
    return 0;
}
