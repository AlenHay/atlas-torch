/* Fortran Fw.d / Iw input of one fixed-width column for every record (synthe/lines/_fixed.py fcol/icol), the same
   algorithm as the numpy reference: BLANK='NULL' (blanks ignored, all-blank = 0), digits accumulated into an
   integer mantissa M, value = M / 10**k (k = digits after the point, or d when there is none) as ONE correctly
   rounded double division, sign applied after.  Fields with any other character, two points/signs, a sign after a
   digit/point, > 15 digits (Fw.d) / > 18 digits (Iw), or a sign/point without digits are flagged in `bad` and left
   to the scalar reader.  Bit-identical to fcol/icol by construction. */
#include <stdint.h>

static const double P10[23] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14,
                               1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

void fcol_c(const uint8_t *a, int64_t n, int64_t stride, int lo, int hi, int d, double *out, uint8_t *bad) {
    for (int64_t r = 0; r < n; r++) {
        const uint8_t *c = a + r * stride;
        int64_t mant = 0;
        int nd = 0, ndot = 0, nsg = 0, neg = 0, kdig = 0, seen = 0, first_num = -1, sgpos = -1, other = 0;
        for (int j = lo; j < hi; j++) {
            uint8_t ch = c[j];
            if (ch >= 48 && ch <= 57) {
                if (nd < 18) mant = mant * 10 + (ch - 48);
                nd++;
                if (seen) kdig++;
                if (first_num < 0) first_num = j;
            } else if (ch == 46) {
                ndot++; seen = 1;
                if (first_num < 0) first_num = j;
            } else if (ch == 45 || ch == 43) {
                nsg++; sgpos = j; if (ch == 45) neg = 1;
            } else if (ch != 32) {
                other = 1;
            }
        }
        int b = other || ndot > 1 || nsg > 1 || (nsg == 1 && first_num >= 0 && sgpos > first_num) || nd > 15 ||
                (nd == 0 && (ndot > 0 || nsg > 0));
        bad[r] = (uint8_t)b;
        if (b) { out[r] = 0.0; continue; }
        int k = ndot > 0 ? kdig : d;
        if (k > 22) k = 22;
        double v = (double)mant / P10[k];
        out[r] = (neg && nd > 0) ? -v : v;
    }
}

void icol_c(const uint8_t *a, int64_t n, int64_t stride, int lo, int hi, int64_t *out, uint8_t *bad) {
    for (int64_t r = 0; r < n; r++) {
        const uint8_t *c = a + r * stride;
        int64_t v = 0;
        int nd = 0, nsg = 0, neg = 0, first_d = -1, sgpos = -1, other = 0;
        for (int j = lo; j < hi; j++) {
            uint8_t ch = c[j];
            if (ch >= 48 && ch <= 57) {
                if (nd < 18) v = v * 10 + (ch - 48);
                nd++;
                if (first_d < 0) first_d = j;
            } else if (ch == 45 || ch == 43) {
                nsg++; sgpos = j; if (ch == 45) neg = 1;
            } else if (ch != 32) {
                other = 1;
            }
        }
        int b = other || nsg > 1 || (nsg == 1 && ((first_d >= 0 && sgpos > first_d) || nd == 0)) || nd > 18;
        bad[r] = (uint8_t)b;
        out[r] = b ? 0 : (neg ? -v : v);
    }
}
