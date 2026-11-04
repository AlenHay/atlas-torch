/* Batched glibc libm calls for the ATLAS9 port (atlas_torch.atlas9.native).
 *
 * Each function applies the *scalar* glibc entry point (exp, log, log10, pow, expf, logf, log10f, powf) to every
 * element of a contiguous array: the results are the very same bits as the Fortran's libm calls and Python's
 * math.exp/... element by element, only without the per-element Python overhead.  Built with -O2 -fno-builtin
 * -fno-fast-math -fno-tree-vectorize so that gcc neither folds nor vectorises the calls (no libmvec variants).
 */
#include <math.h>

void v_exp(const double *x, double *y, long n) { for (long i = 0; i < n; i++) y[i] = exp(x[i]); }
void v_log(const double *x, double *y, long n) { for (long i = 0; i < n; i++) y[i] = log(x[i]); }
void v_log10(const double *x, double *y, long n) { for (long i = 0; i < n; i++) y[i] = log10(x[i]); }
void v_pow(const double *x, const double *p, double *y, long n) { for (long i = 0; i < n; i++) y[i] = pow(x[i], p[i]); }
void v_expf(const float *x, float *y, long n) { for (long i = 0; i < n; i++) y[i] = expf(x[i]); }
void v_logf(const float *x, float *y, long n) { for (long i = 0; i < n; i++) y[i] = logf(x[i]); }
void v_log10f(const float *x, float *y, long n) { for (long i = 0; i < n; i++) y[i] = log10f(x[i]); }
void v_powf(const float *x, const float *p, float *y, long n) { for (long i = 0; i < n; i++) y[i] = powf(x[i], p[i]); }

#ifndef ATLAS9_NATIVE_SHA
#define ATLAS9_NATIVE_SHA ""
#endif
const char *atlas9_native_sha(void) { return ATLAS9_NATIVE_SHA; }
