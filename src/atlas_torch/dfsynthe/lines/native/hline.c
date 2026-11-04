/* DFSYNTHE XLINOP hydrogen line (dfsynthe.for labels 600-634: HPROF4 / HPROF4m / HPROF4p, SOFBET, VCSE1F; every
   point up to the 21st, then every 20th point computed, the 19 between log-interpolated) for one TAPE19 line and
   all layers, REAL*4 operation by operation as the torch port (lines/xlinop.py XLinop._h_wing, lines/hydrogen.py),
   which is the reference and the fallback.  Stock ifx -fp-model fast forms (Flavor.fastmath): Doppler core with
   D=|FREQ-(FINEST+FREQ0)|*(1/DOP) and 4-lane sums, BETA4000=(4000*C)*DBETA/FO, the Lyman-alpha label-52 term as one
   numerator/denominator pair (0/0 = NaN under FTZ, see hydrogen.py _lya_h2plus_fast), the H2+ cutoff 4000-20000
   cm-1 red of Lyman alpha as ((XNFPH(J,2)*K)*DOP)*10**(CUTOFF-14) (no subnormal intermediate, _lya_h2plus_cut_fast);
   else the source forms.

   SOFBET's SAVEd cache: per layer (INDEXSAVE, P of the last recompute) in sidx/sp (carry-in -> carry-out, the
   driver's state machine); a call with B<500 recomputes (P := PP(J)) iff INDEX != INDEXSAVE.  first/change record
   the first INDEX called in a layer and whether an INDEX change happened after it (XLinop._true_carry).

   Transcendentals: LOG/EXP/10.** of REAL*4 arguments as (float)f((double)x) with glibc, which equals the torch
   port's torch.f(x.double()).float() for every float32 argument (exhaustive check, see tests).  FTZ/DAZ: the
   calling thread's MXCSR (ftz.scope).  Build: python -m atlas_torch.dfsynthe.lines.native (or `make native`). */
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {                      /* HPROF4 per-line set-up (Hydrogen.setup) */
    int n, m, index, nfin;
    float y1num, y1wht, freqnm, dbeta, c1con, c2con, radamp, resont, vdw, stark;
    float finest[16], finswt[16];
} hsetup_t;

typedef struct {                      /* per-temperature depth vectors (Hydrogen.__init__) and tables */
    const float *pp, *fo, *y1b, *y1s, *t3nhe, *t3nh2, *c1d, *c2d, *gcon1, *gcon2, *xne, *xnfph1, *xnfph2;
    const float *h0tab; int64_t nh0;
    const float *extab; int64_t nextab;
    const float *cutoffh2, *cutoffh2plus;                  /* 91, 111 */
    const float *propbm;                                   /* [ip 5][j 15][index 7] */
    const float *sc, *sd;                                  /* [ip 5][index 7] */
    const float *spp, *sbeta;                              /* 5, 15 */
    float c52;                                             /* folded constant of the stock label-52 term */
    int fastmath;
} hdepth_t;

typedef struct {                      /* one line */
    int nl, path, do_red;             /* path: 0 alpha (ncon=0 or m=n+1), 1 beta (m=n+2), 2 general */
    int64_t nb, length;
    double wl, wlbeg, w0r, w0b, alphahyd_n, ratio;
    double redcut, wlminus1, wlminus2, bluecut, wlplus1, wlplus2;
    const uint8_t *keep; const float *k0, *dopph;
    const double *wcon, *wtail; const float *k0red, *k0blue;
    const hsetup_t *s_main, *s_red, *s_blue;
    const float *cont; int64_t cont_stride, cont_cols;
    float *buf; int64_t buf_stride; int emit;
    int64_t *sidx; float *sp; int64_t *first; uint8_t *change;
    int64_t stats[3];                 /* h_samples, contribs, extab_oob */
} hline_t;

#define C10 2.99792458E10f
static inline float logf_t(float x) { return (float)log((double)x); }
static inline float expf_t(float x) { return (float)exp((double)x); }
static inline float pow10f_t(float y) { return (float)pow(10.0, (double)y); }
static inline float sqrtf_t(float x) { return sqrtf(x); }

/* SOFBET(B,P,N,M) with the P of its cached tables (p_eff) */
static float sofbet(const hdepth_t *H, float b, float p_eff, int index) {
    float sb = sqrtf_t(b), b2 = b * b;
    if (!(b < 500.f)) return (1.5f / sb + 27.f / b2) / b2;
    if (p_eff != p_eff) p_eff = 0.f;
    int64_t im = (int64_t)truncf(5.f * p_eff) + 1;
    if (im > 4) im = 4;
    int64_t ip = im + 1;
    float wtpp = 5.f * (p_eff - H->spp[im - 1]);
    float wtpm = 1.f - wtpp;
    int ix = index - 1;
    if (!(b < 25.12f)) {
        float cc = H->sc[(ip - 1) * 7 + ix] * wtpp + H->sc[(im - 1) * 7 + ix] * wtpm;
        float dd = H->sd[(ip - 1) * 7 + ix] * wtpp + H->sd[(im - 1) * 7 + ix] * wtpm;
        float corr = 1.f + dd / (cc + b * sb);
        return (1.5f / sb + 27.f / b2) / b2 * corr;
    }
    int jj = 15;
    for (int J = 2; J <= 15; J++) if (b <= H->sbeta[J - 1]) { jj = J; break; }
    int jm = jj - 1, jp = jj;
    float prob_jp = H->propbm[((ip - 1) * 15 + (jp - 1)) * 7 + ix] * wtpp + H->propbm[((im - 1) * 15 + (jp - 1)) * 7 + ix] * wtpm;
    float prob_jm = H->propbm[((ip - 1) * 15 + (jm - 1)) * 7 + ix] * wtpp + H->propbm[((im - 1) * 15 + (jm - 1)) * 7 + ix] * wtpm;
    float wtbp = (b - H->sbeta[jm - 1]) / (H->sbeta[jp - 1] - H->sbeta[jm - 1]);
    float wtbm = 1.f - wtbp;
    float corr = (1.f + prob_jp * wtbp) + prob_jm * wtbm;
    float wt = .5f * (10.f - b);
    wt = wt < 1.f ? wt : 1.f;                                /* clamp(max=1) then clamp(min=0), NaN-propagating */
    wt = wt < 0.f ? 0.f : wt;
    float pr1 = b <= 10.f ? 8.f / (83.f + (2.f + .95f * b2) * b) : 0.f;
    float pr2 = b >= 8.f ? (1.5f / sb + 27.f / b2) / b2 : 0.f;
    return (pr1 * wt + pr2 * (1.f - wt)) * corr;
}

static float vcse1f(float x) {
    if (x <= 0.f) return 0.f;
    if (x <= .01f) return (-logf_t(x) - .577215f) + x;
    if (x <= 1.f)
        return (-logf_t(x) - .57721566f) +
               x * (.99999193f + x * (-.24991055f + x * (.05519968f + x * (-.00976004f + x * .00107857f))));
    if (x <= 30.f)
        return (((x * (x + 2.334733f) + .25062f) / (x * (x + 3.330657f) + 1.681534f)) / x) * expf_t(-x);
    return 0.f;
}

/* one SOFBET call with B<500 goes through the cache state of layer j */
static inline float sofbet_call(const hdepth_t *H, hline_t *L, int j, float b, int index) {
    if (b < 500.f) {
        if (index != L->sidx[j]) {
            if (L->first[j] == 0) L->first[j] = index;
            else L->change[j] = 1;
            L->sidx[j] = index;
            L->sp[j] = H->pp[j];
        } else if (L->first[j] == 0) {
            L->first[j] = index;
        }
    }
    return sofbet(H, b, L->sp[j], index);
}

/* HPROF4 (variant 0), HPROF4m (1: GAM stays GNOT), HPROF4p (2) of layer j at WL0, DELW */
static float hprof4(const hdepth_t *H, hline_t *L, const hsetup_t *s, int variant, int j, double wl0, double delw,
                    float dopph) {
    const int n = s->n, m = s->m;
    const float freqnm = s->freqnm;
    float wl = (float)(wl0 * 10.0 + delw * 10.0);
    float freq = 2.99792458E18f / wl;
    float freq0 = (float)((double)2.99792458E17f / wl0);
    float dl = fabsf(freq - freq0);
    float fo = H->fo[j];
    float hwstk = s->stark * fo;
    float hwvdw = s->vdw * H->t3nhe[j] + (2.f * s->vdw) * H->t3nh2[j];
    float hwrad = s->radamp;
    float hwres = (s->resont * H->xnfph1[j]) * 2.f;
    float hwlor = (hwres + hwvdw) + hwrad;
    float hwdop = dopph;
    int nwid = (hwdop >= hwstk && hwdop >= hwlor) ? 1 : (hwlor >= hwstk ? 2 : 3);
    float mx = (hwdop != hwdop || hwlor != hwlor) ? NAN : (hwdop > hwlor ? hwdop : hwlor);   /* torch.maximum x2 */
    mx = (mx != mx || hwstk != hwstk) ? NAN : (mx > hwstk ? mx : hwstk);
    float hfwid = freqnm * mx;
    int ifcore = fabsf(dl) <= hfwid;
    float dop = freq0 * hwdop;
    float hd = 0.f, hl = 0.f, hs = 0.f, hst = 0.f;
    int need_d = !ifcore || nwid == 1, need_l = !ifcore || nwid == 2, need_s = !ifcore || nwid == 3;
    if (need_d) {                                            /* Doppler core with fine structure */
        int nf = s->nfin;
        if (H->fastmath) {
            float rdop = 1.f / dop;
            float lane[4] = {0.f, 0.f, 0.f, 0.f};
            int nv = nf & ~3;
            for (int i = 0; i < nv; i++) {
                float d = fabsf(freq - (s->finest[i] + freq0)) * rdop;
                if (!(d <= 7.f)) continue;
                int64_t idx = (int64_t)truncf(d * 200.f + 1.5f);
                if (idx < 1) idx = 1;
                if (idx > H->nh0) idx = H->nh0;
                lane[i % 4] = lane[i % 4] + H->h0tab[idx - 1] * s->finswt[i];
            }
            hd = nv ? (lane[1] + lane[3]) + (lane[0] + lane[2]) : 0.f;
            for (int i = nv; i < nf; i++) {
                float d = fabsf(freq - (s->finest[i] + freq0)) * rdop;
                if (!(d <= 7.f)) continue;
                int64_t idx = (int64_t)truncf(d * 200.f + 1.5f);
                if (idx < 1) idx = 1;
                if (idx > H->nh0) idx = H->nh0;
                hd = hd + H->h0tab[idx - 1] * s->finswt[i];
            }
        } else {
            for (int i = 0; i < nf; i++) {
                float d = fabsf((freq - freq0) - s->finest[i]) / dop;
                if (!(d <= 7.f)) continue;
                int64_t idx = (int64_t)truncf(200.f * d + 1.5f);
                if (idx < 1) idx = 1;
                if (idx > H->nh0) idx = H->nh0;
                hd = hd + H->h0tab[idx - 1] * s->finswt[i];
            }
        }
    }
    if (need_l) {                                            /* Lorentz */
        if (n == 1 && m == 2) {
            float hwres4 = hwres * 4.f;
            float hwlor4 = (hwres4 + hwvdw) + hwrad;
            float hhw = freqnm * hwlor4;
            float den = dl * dl + hhw * hhw;
            float hprofres;
            if (freq > (82259.105f - 4000.f) * C10) {
                hprofres = (((hwres4 * freqnm) / 3.14159f) / den * 1.77245f) * dop;
            } else {
                const float spacing = 200.f * C10, f22 = (82259.105f - 22000.f) * C10;
                const float *ch2 = H->cutoffh2;
                float cut;
                if (freq < f22) cut = (ch2[1] - ch2[0]) / spacing * (freq - f22) + ch2[0];
                else {
                    int64_t icut = (int64_t)truncf((freq - f22) / spacing);
                    if (icut > 89) icut = 89;
                    if (icut < 0) icut = 0;
                    float cutfreq = (float)icut * spacing + f22;
                    cut = (ch2[icut + 1] - ch2[icut]) / spacing * (freq - cutfreq) + ch2[icut];
                }
                cut = ((pow10f_t(cut - 14.f) * H->xnfph1[j]) * 2.f) / C10;
                if (freq < 50000.f * C10) cut = 0.f;
                hprofres = (cut * 1.77245f) * dop;
            }
            float hprofrad = 0.f;
            if (freq > 2.4190611E15f && freq < .77f * 3.28805E15f)
                hprofrad = (((hwrad * freqnm) / 3.14159f) / den * 1.77245f) * dop;
            float hprofvdw = (((hwvdw * freqnm) / 3.14159f) / den * 1.77245f) * dop;
            if (freq < 1.8E15f) hprofvdw = 0.f;
            hl = (hprofres + hprofrad) + hprofvdw;
        } else {
            float hhw = freqnm * hwlor;
            float top = hhw;
            if (n == 1 && m <= 5) {
                float a1 = 0, a2 = 0;
                int win = 1;
                if (m == 3) { a1 = .885f * 3.288051E15f; a2 = .890f * 3.288051E15f; }
                else if (m == 4) { a1 = .936f * 3.288051E15f; a2 = .938f * 3.288051E15f; }
                else if (m == 5) { a1 = .959f * 3.288051E15f; a2 = .961f * 3.288051E15f; }
                else win = 0;
                if (win && freq > a1 && freq < a2) top = hhw - freqnm * hwrad;
            }
            hl = ((top / 3.14159f) / (dl * dl + hhw * hhw) * 1.77245f) * dop;
        }
    }
    if (need_s) {                                            /* Stark (SOFBET calls) */
        float wty1 = 1.f / (1.f + H->xne[j] / s->y1wht);
        float y1scal = (s->y1num * H->y1s[j]) * wty1 + H->y1b[j] * (1.f - wty1);
        float c1 = (H->c1d[j] * s->c1con) * y1scal;
        float c2 = H->c2d[j] * s->c2con;
        float beta = (fabsf(dl) / fo) * s->dbeta;
        float dbeta = s->dbeta;
        float g1 = 6.77f * sqrtf_t(c1);
        float lg = .2114f + logf_t(sqrtf_t(c2) / c1);
        lg = lg < 0.f ? 0.f : lg;                            /* clamp(min=0), NaN propagates */
        float gnot = (g1 * lg) * ((1.f - H->gcon1[j]) - H->gcon2[j]);
        float y1 = c1 * beta;
        float y2 = c2 * (beta * beta);
        int skip = (y2 <= 1.E-4f) && (y1 <= 1.E-5f);
        float y1c = y1 > 80.f ? 80.f : y1;
        double x100 = (double)(100.f * y1c);
        int64_t ni;
        if (x100 != x100) ni = INT64_MIN;
        else { double a = floor(fabs(x100) + 0.5); ni = (int64_t)(x100 < 0 ? -a : a); }
        if (ni < 1 && !skip) L->stats[2]++;
        float ex = 0.f;
        if (ni >= 1) { int64_t k = ni - 1; if (k > H->nextab - 1) k = H->nextab - 1; ex = H->extab[k]; }
        float t90 = 90.f * y1;
        float p3 = t90 * (t90 * t90);
        float gam = g1 * ((0.5f * ex + vcse1f(y1)) - 0.5f * vcse1f(y2)) *
                    ((1.f - H->gcon1[j] / (1.f + p3)) - H->gcon2[j] / (1.f + 2000.f * y1));
        if (variant == 1) gam = gnot;
        if (gam <= 1.E-20f) gam = 0.f;
        if (skip) gam = gnot;
        float prqs = sofbet_call(H, L, j, beta, s->index);
        if (m <= 2) {
            prqs = prqs * 0.5f;
            const float f20 = (82259.105f - 20000.f) * C10, f4000 = (82259.105f - 4000.f) * C10;
            if (freq < f20) hs = 0.f;
            else if (freq > f4000) {
                float beta4000 = H->fastmath ? ((4000.f * C10) * dbeta) / fo : ((4000.f * C10) / fo) * dbeta;
                float sb = sofbet_call(H, L, j, beta4000, s->index);
                if (H->fastmath) {
                    float a = (H->c52 * dop) * dbeta;
                    float num = (fo * a) * (H->xnfph2[j] * prqs);
                    float dn = (fo * dbeta) * (sb * 0.5f);
                    hs = num / dn;
                } else {
                    float prqsp4000 = ((sb * 0.5f) / fo) * dbeta;
                    float cut4000 = (pow10f_t(-11.07f - 14.f) / C10) * H->xnfph2[j];
                    hs = (((((cut4000 / prqsp4000) * prqs) / fo) * dbeta) * 1.77245f) * dop;
                }
            } else {
                const float f15 = (82259.105f - 15000.f) * C10, spacing = 100.f * C10;
                const float *ch = H->cutoffh2plus;
                float cut;
                if (freq < f15) cut = (ch[1] - ch[0]) / spacing * (freq - f15) + ch[0];
                else {
                    int64_t icut = (int64_t)truncf((freq - f15) / spacing);
                    if (icut > 109) icut = 109;
                    if (icut < 0) icut = 0;
                    float cutfreq = (float)icut * spacing + f15;
                    cut = (ch[icut + 1] - ch[icut]) / spacing * (freq - cutfreq) + ch[icut];
                }
                if (H->fastmath) {                           /* stock: ((XNFPH(J,2)*K)*DOP)*10**(CUTOFF-14) */
                    const float k = (1.f / C10) * 1.77245f;     /* folded, 0x2E82030B (hydrogen.py _kh2plus) */
                    hs = ((H->xnfph2[j] * k) * dop) * pow10f_t(cut - 14.f);
                } else {
                    cut = (pow10f_t(cut - 14.f) / C10) * H->xnfph2[j];
                    hs = (cut * 1.77245f) * dop;
                }
            }
        }
        float F = gam > 0.f ? (gam / 3.14159f) / (gam * gam + beta * beta) : 0.f;
        float p1 = (.9f * y1) * (.9f * y1);
        float fns = (p1 + .03f * sqrtf_t(y1)) / (p1 + 1.f);
        hst = ((((prqs * (1.f + fns) + F) / fo) * dbeta) * 1.77245f) * dop;
    }
    if (ifcore) {
        if (nwid == 1) return 0.f + hd;
        if (nwid == 2) return 0.f + hl;
        return (0.f + hs) + hst;
    }
    return (((0.f + hd) + hl) + hs) + hst;
}

static inline float contv(const hline_t *L, int j, int64_t bin) {     /* CONTINUUM(bin), bin 1-based, clamped */
    int64_t k = bin - 1;
    if (k < 0) k = 0;
    if (k > L->length - 1) k = L->length - 1;
    return L->cont[j * L->cont_stride + k];
}

static inline void addbin(hline_t *L, int j, int64_t bin, float v) {
    if (L->emit) { float *p = L->buf + j * L->buf_stride + (bin - 1); *p = *p + v; L->stats[1]++; }
}

/* one wing of layer j: step +1 red over bins start..start+count-1, step -1 blue over bins start-1 down (count) */
static void wing(const hdepth_t *H, hline_t *L, int j, int step, int64_t start, int64_t count) {
    const int general = L->path == 2;
    const hsetup_t *sm = L->s_main, *ss = step > 0 ? L->s_red : L->s_blue;
    const float k0 = L->k0[j], dopph = L->dopph[j];
    const double wl = L->wl, ratio = L->ratio;
    const double wcon = general ? L->wcon[j] : 0., wtail = general ? L->wtail[j] : 0.;
    double wave = step > 0 ? L->w0r : L->w0b;
    float oldk = 0.f;
    int64_t cnt = 0;                                         /* IB (red) / I (blue) */
    for (int64_t off = 0; off < count; off++) {
        int64_t bin;
        if (step > 0) { bin = start + off; if (off) wave = wave * ratio; }
        else { bin = start - 1 - off; wave = wave / ratio; }
        if (step > 0) {
            if (general) { if (wave < wcon) continue; if (wave > L->wlminus1) break; }
            if (L->path == 1 && wave > L->alphahyd_n) break;
        } else if (general) {
            if (wave < wcon || wave < L->wlplus1) break;
        }
        cnt++;
        if (cnt > 21 && cnt % 20 != 1) continue;
        L->stats[0]++;
        float kap = k0 * hprof4(H, L, sm, 0, j, wl, wave - wl, dopph);
        if (general) {
            if (wave < wtail) kap = (float)(((double)kap * (wave - wcon)) / (wtail - wcon));
            if (step > 0 ? wave > L->redcut : wave < L->bluecut) {
                double ws = step > 0 ? L->wlminus2 : L->wlplus2;
                float kr = (step > 0 ? L->k0red[j] : L->k0blue[j]) *
                           hprof4(H, L, ss, step > 0 ? 1 : 2, j, wl, wave - ws, dopph);
                if (wave < wtail) kr = (float)(((double)kr * (wave - wcon)) / (wtail - wcon));
                if (step > 0 ? kr >= kap : kr > kap) break;      /* GO TO 612 / 900 without adding */
            }
        }
        addbin(L, j, bin, kap);
        float newk = logf_t(kap > 1.E-20f ? kap : 1.E-20f);     /* MAX(KAPPA,1.E-20) = maxss: NaN -> 1.E-20 */
        if (cnt > 21) {
            float dk = (newk - oldk) / 20.f;
            for (int k = 1; k <= 19; k++) addbin(L, j, bin - step * k, expf_t(newk - (float)k * dk));
        }
        oldk = newk;
        if (kap < contv(L, j, bin)) break;
    }
}

/* the hydrogen line for every layer with keep[j]: red wing (if do_red), then blue unless MINRED=1 or WL<WLBEG */
int h_line(const hdepth_t *H, hline_t *L) {
    for (int j = 0; j < L->nl; j++) {
        if (!L->keep[j]) continue;
        int64_t minred = L->nb > 1 ? L->nb : 1;
        if (L->do_red) {
            wing(H, L, j, +1, minred, L->length - minred + 1);
            if (minred == 1 || L->wl < L->wlbeg) continue;
        }
        int64_t ib0 = L->nb < L->length + 1 ? L->nb : L->length + 1;
        wing(H, L, j, -1, ib0, ib0 - 1);
    }
    return 0;
}

/* the kernel's transcendentals on an array (tests: equality with the torch float64 path) */
void hl_transcendental(int which, const float *x, int64_t n, float *out) {
    for (int64_t i = 0; i < n; i++)
        out[i] = which == 0 ? logf_t(x[i]) : (which == 1 ? expf_t(x[i]) : pow10f_t(x[i]));
}
