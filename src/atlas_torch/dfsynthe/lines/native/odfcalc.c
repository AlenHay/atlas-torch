/* DFSYNTHE DFCALC (dfsynthe.for) for several spectra: IBUFFER = NINT(1000*LOG10(BUFFER)) (the REAL*4 products
   1000*LOG10 come from the caller), per interval IDF the BISORT of IBUFFER+30000 and the 12 integer step averages
   ISTEPS = (ISUM + N/2)/N over the rank ranges NSTEPS(ISTEP..ISTEP+1, IDF).

   Only the sums of the sorted keys between fixed ranks are used.  When every key of an interval is in
   [0, 65535], BISORT (a stable LSD radix sort on the bit length of MAX) orders the keys by value, so any value
   sort gives the same sums: a 2-pass 8-bit counting sort is used.  Otherwise the Fortran BISORT is run literally.
   Integer arithmetic only: exact, bit-identical to dfsynthe.dfcalc (torch/numpy reference).
   Build: python -m atlas_torch.dfsynthe.lines.native (or `make native`). */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NIDF 1540

static void bisort_literal(int32_t *k, int32_t *k1, int64_t n) {    /* keys already +30000 */
    int32_t mask = 1, mx = k[0];
    for (int64_t j = 1; j < n; j++) mx = mx > k[j] ? mx : k[j];
    for (int i = 1; i <= 59; i++) {
        if (i != 1) { mx = mx / 2; mask = mask * 2; }
        if (mx == 0) return;
        int64_t j1 = 0, j0 = 0;
        for (int64_t j = 0; j < n; j++) {
            if ((k[j] & mask) > 0) k1[j1++] = k[j];
            else k[j0++] = k[j];
        }
        for (int64_t j = 0; j < j1; j++) k[j0 + j] = k1[j];
    }
}

static void radix16(int32_t *k, int32_t *tmp, int64_t n) {           /* keys in [0, 65535] */
    int64_t c[256];
    for (int pass = 0; pass < 2; pass++) {
        int sh = 8 * pass;
        memset(c, 0, sizeof c);
        for (int64_t j = 0; j < n; j++) c[(k[j] >> sh) & 255]++;
        int64_t s = 0;
        for (int b = 0; b < 256; b++) { int64_t t = c[b]; c[b] = s; s += t; }
        for (int64_t j = 0; j < n; j++) tmp[c[(k[j] >> sh) & 255]++] = k[j];
        memcpy(k, tmp, (size_t)n * sizeof *k);
    }
}

/* v: [nrows, row_stride] float32 = LOG10(BUFFER)*1000.; nbeg/nend: 1-based NBEG/NEND(1540); nsteps: [13][1540];
   out: [nrows][1540][12] int16.  Returns 0, or -1 (nothing written for that row and after) on a non-finite
   value, for which the caller uses its reference path. */
int dfcalc_rows(int nrows, int64_t length, int64_t row_stride, const float *v, const int64_t *nbeg,
                const int64_t *nend, const int64_t *nsteps, int16_t *out) {
    int64_t maxn = 0;
    for (int i = 0; i < NIDF; i++) if (nend[i] - nbeg[i] + 1 > maxn) maxn = nend[i] - nbeg[i] + 1;
    int32_t *ib = malloc((size_t)length * sizeof *ib);
    int32_t *k = malloc((size_t)maxn * sizeof *k), *k1 = malloc((size_t)maxn * sizeof *k1);
    int32_t *hist = malloc(65536 * sizeof *hist);
    if (!ib || !k || !k1 || !hist) { free(ib); free(k); free(k1); free(hist); return -2; }
    for (int idf = 0; idf < NIDF; idf++)                   /* the prefix walk needs ascending NSTEPS */
        for (int t = 0; t < 12; t++)
            if (nsteps[(t + 1) * NIDF + idf] < nsteps[t * NIDF + idf]) { free(ib); free(k); free(k1); free(hist); return -3; }
    int rc = 0;
    for (int r = 0; r < nrows && rc == 0; r++) {
        const float *x = v + (int64_t)r * row_stride;
        for (int64_t i = 0; i < length; i++) {
            double d = (double)x[i];
            if (!isfinite(d) || fabs(d) > 2.0e9) { rc = -1; break; }
            double a = floor(fabs(d) + 0.5);
            ib[i] = (int32_t)(d < 0 ? -a : a);                         /* NINT */
        }
        if (rc) break;
        for (int idf = 0; idf < NIDF; idf++) {
            int64_t b = nbeg[idf], e = nend[idf], n = e - b + 1;
            int ok = 1;
            for (int64_t i = 0; i < n; i++) {
                int32_t kk = ib[b - 1 + i] + 30000;
                k[i] = kk;
                ok &= (kk >= 0) & (kk <= 65535);
            }
            int16_t *o = out + ((int64_t)r * NIDF + idf) * 12;
            int64_t S[13];                      /* S[s] = sum of the NSTEPS(s+1)-1 smallest (key-30000) */
            int32_t mn = 65535, mx = 0;
            if (ok) for (int64_t i = 0; i < n; i++) { if (k[i] < mn) mn = k[i]; if (k[i] > mx) mx = k[i]; }
            int64_t w = (int64_t)mx - mn + 1;
            if (ok && n > 0 && w >= 1 && w <= 65536 && w <= 4 * n + 1024) {
                /* histogram: the sorted sequence is implied by the counts (equal keys are interchangeable) */
                memset(hist, 0, (size_t)w * sizeof *hist);
                for (int64_t i = 0; i < n; i++) hist[k[i] - mn]++;
                int64_t c = 0, sum = 0, v = 0;
                for (int t = 0; t < 13; t++) {
                    int64_t pr = nsteps[t * NIDF + idf] - 1;           /* prefix length */
                    while (v < w && c + hist[v] <= pr) { c += hist[v]; sum += (int64_t)hist[v] * (v + mn - 30000); v++; }
                    S[t] = sum + (pr - c) * (v < w ? (int64_t)(v + mn - 30000) : 0);
                }
            } else {
                if (ok) radix16(k, k1, n);
                else bisort_literal(k, k1, n);
                int64_t sum = 0, pos = 0;
                for (int t = 0; t < 13; t++) {
                    int64_t pr = nsteps[t * NIDF + idf] - 1;
                    for (; pos < pr; pos++) sum += (int64_t)k[pos] - 30000;
                    S[t] = sum;
                }
            }
            for (int s = 0; s < 12; s++) {
                int64_t a0 = nsteps[s * NIDF + idf], a1 = nsteps[(s + 1) * NIDF + idf];
                int64_t isum = S[s + 1] - S[s];
                int64_t nn = a1 - a0;
                if (nn == 0) { o[s] = 0; continue; }
                int64_t t = isum + nn / 2;
                int64_t q = (t < 0 ? -t : t) / nn;
                o[s] = (int16_t)(t < 0 ? -q : q);
            }
        }
    }
    free(ib); free(k); free(k1); free(hist);
    return rc;
}

/* Turbulent broadening of the saved 0 km/s spectrum (MAIN labels 381-420), as dfsynthe.broaden: for every bin b
   (0-based) OUT(b) = B(b)*VTCENTER + sum_{d=-NVT..-1,1..NVT} B(b+d)*VTPROF(|d|) in that order (B = 0 below
   bin 0), except the first NVT bins of records 2.. (RECLEN bins each), whose centre term comes after the red terms
   of the previous record (kc = NVT - offset in record terms first).  REAL*4 products and sums, no FMA. */
#define RECLEN 10000
#define CH 4096
int broaden_rows(int nl, int64_t lenbuff, int64_t length, int nvt, float vtc, const float *vtp, const float *b0,
                 float *out) {
    float tmp[CH];
    for (int r = 0; r < nl; r++) {
        const float *B = b0 + (int64_t)r * lenbuff;
        float *O = out + (int64_t)r * length;
        if (length + nvt > lenbuff) return -1;
        for (int64_t s0 = 0; s0 < length; s0 += CH) {
            int64_t e = s0 + CH < length ? s0 + CH : length, n = e - s0;
            for (int64_t i = 0; i < n; i++) tmp[i] = B[s0 + i] * vtc;
            for (int d = -nvt; d <= nvt; d++) {
                if (d == 0) continue;
                float w = vtp[(d < 0 ? -d : d) - 1];
                int64_t i0 = 0;
                for (; i0 < n && s0 + i0 + d < 0; i0++) tmp[i0] = tmp[i0] + 0.f * w;
                const float *src = B + s0 + d;
                for (int64_t i = i0; i < n; i++) tmp[i] = tmp[i] + src[i] * w;
            }
            for (int64_t i = 0; i < n; i++) O[s0 + i] = tmp[i];
        }
        for (int64_t rs = RECLEN + 1; rs <= length; rs += RECLEN) {   /* 1-based record starts 2.. */
            for (int t0 = 0; t0 < nvt && rs + t0 <= length; t0++) {
                int64_t b = rs + t0;                                       /* 1-based bin */
                int kc = nvt - (int)((b - 1) % RECLEN);
                float acc = 0.f;
                for (int t = 0; t <= 2 * nvt; t++) {
                    int d; float w;
                    if (t == kc) { d = 0; w = vtc; }
                    else {
                        int li = t < kc ? t : t - 1;                       /* lst = -NVT..-1, 1..NVT */
                        d = li < nvt ? li - nvt : li - nvt + 1;
                        w = li < nvt ? vtp[nvt - 1 - li] : vtp[li - nvt];
                    }
                    acc = acc + B[b + d - 1] * w;
                }
                O[b - 1] = acc;
            }
        }
    }
    return 0;
}
