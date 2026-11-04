/* JOSH's Gauss-Seidel sweeps (atlas9mem.for, REAL*4), a literal transcription of the numpy reference
 * atlas_torch.atlas9.josh._gs_numpy, one independent problem at a time:
 *   DO K=NXTAU,1,-1: DELXS=(SUM_M COEFJ(K,M)*XS(M) [sequential REAL*4 sum] *XALPHA(K)+XSBAR(K)-XS(K))/DIAG(K);
 *   IFERR if |DELXS/XS(K)| > TOL; XS(K)=MAX(XS(K)+DELXS, 1E-38) (in double, NaN propagating, then REAL*4);
 *   up to NSWEEP sweeps, stopping after the first sweep without IFERR.
 * float arithmetic only (FLT_EVAL_METHOD 0), no contraction (-ffp-contract=off).
 * Arrays are per problem rows: xs, xsbar, xalpha, diag [B][nx]; coefj [nx][nx] row-major (coefj[k*nx+m]).
 */
#include <math.h>

void josh_gs(int nx, long B, const float *coefj, float *xs, const float *xsbar, const float *xalpha,
             const float *diag, const unsigned char *active, int nsweep, float tol)
{
    for (long b = 0; b < B; b++) {
        if (!active[b]) continue;
        float *x = xs + b * nx;
        const float *sb = xsbar + b * nx, *al = xalpha + b * nx, *dg = diag + b * nx;
        for (int sweep = 0; sweep < nsweep; sweep++) {
            int iferr = 0;
            for (int k = nx - 1; k >= 0; k--) {
                const float *c = coefj + (long)k * nx;
                float acc = 0.0f;
                for (int m = 0; m < nx; m++) {
                    float p = c[m] * x[m];
                    acc = acc + p;
                }
                float delxs = (acc * al[k] + sb[k] - x[k]) / dg[k];
                float errorx = fabsf(delxs / x[k]);
                if (errorx > tol) iferr = 1;
                float s = x[k] + delxs;
                double t = (double)s;
                if (!isnan(t) && t < 1e-38) t = 1e-38;
                x[k] = (float)t;
            }
            if (!iferr) break;
        }
    }
}
