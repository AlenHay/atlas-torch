/* Map a scalar math function (a libimf entry point resolved by the caller through ctypes) over an array, so that
   the stock flavour can evaluate libimf expf/exp on millions of SELECTLINES arguments (lines/imf.py).  MXCSR
   (FTZ/DAZ) is inherited from the calling thread, as in the stock executable. */
typedef float (*f1_t)(float);
typedef double (*d1_t)(double);

void map_f1(f1_t f, const float *x, float *y, long n)
{
    for (long i = 0; i < n; i++)
        y[i] = f(x[i]);
}

void map_d1(d1_t f, const double *x, double *y, long n)
{
    for (long i = 0; i < n; i++)
        y[i] = f(x[i]);
}

/* Intel SVML vector entry points (__svml_expf4, __svml_powf4, ...: one SSE register in, one out; standard x86-64
   calling convention), applied 4 lanes at a time; the last partial group is padded with its last element.  The
   stock dfsynthe.exe calls them in its vectorised setup loops, and their lanes can differ from the scalar
   __svml_*1 variants (lines/svml.py). */
#include <xmmintrin.h>
typedef __m128 (*v4f1_t)(__m128);
typedef __m128 (*v4f2_t)(__m128, __m128);

void map_v4f1(v4f1_t f, const float *x, float *y, long n)
{
    for (long i = 0; i < n; i += 4) {
        float t[4], r[4];
        for (int k = 0; k < 4; k++) t[k] = x[i + k < n ? i + k : n - 1];
        _mm_storeu_ps(r, f(_mm_loadu_ps(t)));
        for (int k = 0; k < 4 && i + k < n; k++) y[i + k] = r[k];
    }
}

void map_v4f2(v4f2_t f, const float *x1, const float *x2, float *y, long n)
{
    for (long i = 0; i < n; i += 4) {
        float t1[4], t2[4], r[4];
        for (int k = 0; k < 4; k++) {
            long m = i + k < n ? i + k : n - 1;
            t1[k] = x1[m];
            t2[k] = x2[m];
        }
        _mm_storeu_ps(r, f(_mm_loadu_ps(t1), _mm_loadu_ps(t2)));
        for (int k = 0; k < 4 && i + k < n; k++) y[i + k] = r[k];
    }
}

typedef float (*f2_t)(float, float);

void map_f2(f2_t f, const float *x1, const float *x2, float *y, long n)
{
    for (long i = 0; i < n; i++)
        y[i] = f(x1[i], x2[i]);
}
