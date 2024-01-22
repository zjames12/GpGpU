#ifndef COVMATRIX_FUNS_CUH
#define COVMATRIX_FUNS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#define EXPONENTIAL_ISOTROPIC 0
#define EXPONENTIAL_SCALEDIM 1

__device__ void exponential_isotropic(double* covparms, double* locsub, double* covmat,
                int dim, int m) {
    double temp = 0;
    for (int i1 = 0; i1 < m; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            double d = 0.0;
            for (int j = 0; j < dim; j++) {
                temp = (locsub[i1 * dim + j] - locsub[i2 * dim + j]) / covparms[1];
                d += temp * temp;
            }
            d = sqrt(d);
            // calculate covariance
            if (i1 == i2) {
                covmat[i2 * m + i1] = covparms[0] * (1 + covparms[2]);
            }
            else {
                covmat[i2 * m + i1] = covparms[0] * exp(-d);
                covmat[i1 * m + i2] = covmat[i2 * m + i1];
            }
        }
    }
}

__device__ void d_exponential_isotropic(double* covparms, double* locsub, double* dcovmat,
                int dim, int m, int nparms) {
    double temp = 0;
    for (int i1 = 0; i1 < m; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            dcovmat[i1 * m * nparms + i2 * nparms + 0] = 0;
            dcovmat[i1 * m * nparms + i2 * nparms + 1] = 0;
            double d = 0.0;
            double a = 0;
            for (int j = 0; j < dim; j++) {
                a = (locsub[i1 * dim + j] - locsub[i2 * dim + j]) / covparms[1];
                d += a * a;
            }
            d = sqrt(d);
            temp = exp(-d);

            dcovmat[i1 * m * nparms + i2 * nparms + 0] += temp;
            dcovmat[i1 * m * nparms + i2 * nparms + 1] += covparms[0] * temp * d / covparms[1];
            if (i1 == i2) { // update diagonal entry
                dcovmat[i1 * m * nparms + i2 * nparms + 0] += covparms[2];
                dcovmat[i1 * m * nparms + i2 * nparms + 2] = covparms[0];
            }
            else { // fill in opposite entry
                for (int j = 0; j < nparms; j++) {
                    dcovmat[i2 * m * nparms + i1 * nparms + j] = dcovmat[i1 * m * nparms + i2 * nparms + j];
                }
            }
        }
    }
}

__device__ void exponential_scaledim(double* covparms, double* locsub, double* covmat,
                int dim, int m) {
    double temp = 0;
    for (int i1 = 0; i1 < m; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            double d = 0.0;
            for (int j = 0; j < dim; j++) {
                temp = (locsub[i1 * dim + j] - locsub[i2 * dim + j]) / covparms[1+j];
                d += temp * temp;
            }
            d = sqrt(d);
            if (d == 0.0) {
                covmat[i2 * m + i1] = covparms[0];
            } else {
                covmat[i2 * m + i1] = covparms[0] * exp(-d);
            }
            if (i1 == i2) {
                covmat[i2 * m + i2] += covparms[0] * covparms[dim + 1];
            } else {
                covmat[i1 * m + i2] = covmat[i2 * m + i1];
            }

            // // calculate covariance
            // if (i1 == i2) {
            //     covmat[i2 * m + i1] = covparms[0] * (exp(-d) + covparms[dim + 1]);
            // }
            // else {
            //     covmat[i2 * m + i1] = covparms[0] * exp(-d);
            //     covmat[i1 * m + i2] = covmat[i2 * m + i1];
            // }
        }
    }
}

__device__ void d_exponential_scaledim(double* covparms, double* locsub, double* dcovmat,
                int dim, int m, int nparms) {
    double temp = 0;
    for (int i1 = 0; i1 < m; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            for (int k = 0; k < nparms; k++) {
                dcovmat[i1 * m * nparms + i2 * nparms + k] = 0;
            }
            // dcovmat[i1 * m * nparms + i2 * nparms + 0] = 0;
            // dcovmat[i1 * m * nparms + i2 * nparms + 1] = 0;
            double d = 0.0;
            double a = 0;
            for (int j = 0; j < dim; j++) {
                a = (locsub[i1 * dim + j] - locsub[i2 * dim + j]) / covparms[1 + j];
                d += a * a;
            }
            d = sqrt(d);
            double cov;
            if (d == 0.0) {
                cov = covparms[0];
                dcovmat[i1 * m * nparms + i2 * nparms + 0] += 1;
            } else {
                cov = covparms[0] * exp(-d);
                dcovmat[i1 * m * nparms + i2 * nparms + 0] += cov / covparms[0];
                for (int j = 0; j < dim; j++) {
                    double dj2 = (locsub[i1 * dim + j] - locsub[i2 * dim + j]) / covparms[1 + j];
                    dj2 = dj2 * dj2;
                    dcovmat[i1 * m * nparms + i2 * nparms + j+1] += cov/d*dj2/covparms[j+1];
                }
            }

            if (i1 == i2){
                dcovmat[i1 * m * nparms + i2 * nparms + 0] += covparms[dim+1];
                dcovmat[i1 * m * nparms + i2 * nparms + dim+1] += covparms[0];
            } else {
                for (int j = 0; j < nparms; j++) {
                    dcovmat[i2 * m * nparms + i1 * nparms + j] = dcovmat[i1 * m * nparms + i2 * nparms + j];
                }
            }

        }
    }
}
#endif