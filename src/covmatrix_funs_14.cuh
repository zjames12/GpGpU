#ifndef COVMATRIX_FUNS_CUH
#define COVMATRIX_FUNS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#define EXPONENTIAL_ISOTROPIC 0
#define EXPONENTIAL_SCALEDIM 1
#define EXPONENTIAL_SPACETIME 2
#define EXPONENTIAL_SPHERETIME 3

#define M_PI  3.14159265358979323846 // pi




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
            dcovmat[i1 * m * nparms + i2 * nparms + 2] = 0;
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

__device__ void exponential_spacetime(double* covparms, double* locsub, double* covmat,
                int dim, int m) {
    
    
    double locs_scaled[31 * 4];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dim - 1; j++) {
            locs_scaled[i * dim + j] = locsub[i * dim + j] / covparms[1];
        }
        locs_scaled[i * dim + dim - 1] = locsub[i * dim + dim - 1] / covparms[2];
    }

    

    double newparams[3];

    newparams[0] = covparms[0];
    newparams[1] = 1;
    newparams[2] = covparms[3];
    // exponential_isotropic(newparams, locs_scaled, covmat, dim, m);
    double temp = 0;
    for (int i1 = 0; i1 < m; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            double d = 0.0;
            for (int j = 0; j < dim; j++) {
                temp = (locs_scaled[i1 * dim + j] - locs_scaled[i2 * dim + j]) / newparams[1];
                d += temp * temp;
            }
            d = sqrt(d);
            // calculate covariance
            if (i1 == i2) {
                covmat[i2 * m + i1] = newparams[0] * (1 + newparams[2]);
            }
            else {
                covmat[i2 * m + i1] = newparams[0] * exp(-d);
                covmat[i1 * m + i2] = covmat[i2 * m + i1];
            }
        }
    }
}

__device__ void d_exponential_spacetime(double* covparms, double* locsub, double* dcovmat,
                int dim, int m, int nparms) {
    
    double locs_scaled[31 * 4];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dim - 1; j++) {
            locs_scaled[i * dim + j] = locsub[i * dim + j] / covparms[1];
        }
        locs_scaled[i * dim + dim - 1] = locsub[i * dim + dim - 1] / covparms[2];
    }

    for (int i2 = 0; i2 < m; i2++) {
        for (int i1 = 0; i1 <= i2; i1++) {
            for (int k = 0; k < nparms; k++) {
                dcovmat[i1 * m * nparms + i2 * nparms + k] = 0;
            }
            
            double d = 0.0;
            double a = 0;
            for (int j = 0; j < dim; j++) {
                a = (locs_scaled[i1 * dim + j] - locs_scaled[i2 * dim + j]);
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
                double dj2 = 0;
                for (int j = 0; j < dim - 1; j++) {
                    a = (locs_scaled[i1 * dim + j] - locs_scaled[i2 * dim + j]);
                    dj2 += a * a;
                }
                dcovmat[i1 * m * nparms + i2 * nparms + 1] += cov/d*dj2/covparms[1];
                a = (locs_scaled[i1 * dim + dim - 1] - locs_scaled[i2 * dim + dim - 1]);
                dj2 = a * a;
                dcovmat[i1 * m * nparms + i2 * nparms + 2] += cov/d*dj2/covparms[2];
            }

            if (i1 == i2){
                dcovmat[i1 * m * nparms + i2 * nparms + 0] += covparms[3];
                dcovmat[i1 * m * nparms + i2 * nparms + 3] += covparms[0];
            } else {
                for (int j = 0; j < nparms; j++) {
                    dcovmat[i2 * m * nparms + i1 * nparms + j] = dcovmat[i1 * m * nparms + i2 * nparms + j];
                }
            }
        }
    }
}

__device__ void exponential_spheretime(double* covparms, double* locsub, double* covmat, int dim, int m) {
    double locs[31 * 4];
    for (int i=0; i<m; i++) {
        double lonrad = 2*M_PI*locsub[i * 3 + 0]/360;
        double latrad = 2*M_PI*(locsub[i * 3 + 1]+90)/360;
        locs[i * 4 + 0] = sin(latrad)*cos(lonrad);         // convert lon,lat to x,y,z
        locs[i * 4 + 1] = sin(latrad)*sin(lonrad);
        locs[i * 4 + 2] = cos(latrad);
        
        locs[i * 4 + 3] = locsub[i * 3 + 2];
    }

    dim = 4;

    // exponential_spacetime(covparms, locs, covmat, dim, m);

    double locs_scaled[31 * 4];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dim - 1; j++) {
            locs_scaled[i * dim + j] = locs[i * dim + j] / covparms[1];
        }
        locs_scaled[i * dim + dim - 1] = locs[i * dim + dim - 1] / covparms[2];
    }

    

    double newparams[3];

    newparams[0] = covparms[0];
    newparams[1] = 1;
    newparams[2] = covparms[3];
    // exponential_isotropic(newparams, locs_scaled, covmat, dim, m);
    double temp = 0;
    for (int i1 = 0; i1 < m; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            double d = 0.0;
            for (int j = 0; j < dim; j++) {
                temp = (locs_scaled[i1 * dim + j] - locs_scaled[i2 * dim + j]) / newparams[1];
                d += temp * temp;
            }
            d = sqrt(d);
            // calculate covariance
            if (i1 == i2) {
                covmat[i2 * m + i1] = newparams[0] * (1 + newparams[2]);
            }
            else {
                covmat[i2 * m + i1] = newparams[0] * exp(-d);
                covmat[i1 * m + i2] = covmat[i2 * m + i1];
            }
        }
    }

}

__device__ void d_exponential_spheretime(double* covparms, double* locsub, double* dcovmat,
                int dim, int m, int nparms) {
    
    double locs[31 * 4];
    for (int i=0; i<m; i++) {
        double lonrad = 2*M_PI*locsub[i * 3 + 0]/360;
        double latrad = 2*M_PI*(locsub[i * 3 + 1]+90)/360;
        locs[i * 4 + 0] = sin(latrad)*cos(lonrad);         // convert lon,lat to x,y,z
        locs[i * 4 + 1] = sin(latrad)*sin(lonrad);
        locs[i * 4 + 2] = cos(latrad);
        
        locs[i * 4 + 3] = locsub[i * 3 + 2];
    }

    dim = 4;

    d_exponential_spacetime(covparms, locs, dcovmat, dim, m, nparms);
}
    

__device__ void covariance_func(short covfun_name, double* covparms, double* locsub, double* covmat, int dim, int m) {
    if (covfun_name == EXPONENTIAL_ISOTROPIC) {
        exponential_isotropic(covparms, locsub, covmat, dim, m);
    } else if (covfun_name == EXPONENTIAL_SCALEDIM) {
        exponential_scaledim(covparms, locsub, covmat, dim, m);
    } else if (covfun_name == EXPONENTIAL_SPACETIME) {
        exponential_spacetime(covparms, locsub, covmat, dim, m);
    } else if (covfun_name == EXPONENTIAL_SPHERETIME) {
        exponential_spheretime(covparms, locsub, covmat, dim, m);
    }
}

__device__ void d_covariance_func(short covfun_name, double* covparms, double* locsub, double* dcovmat, int dim, int m, int nparms) {
    if (covfun_name == EXPONENTIAL_ISOTROPIC) {
        d_exponential_isotropic(covparms, locsub, dcovmat, dim, m, nparms);
    } else if (covfun_name == EXPONENTIAL_SCALEDIM) {
        d_exponential_scaledim(covparms, locsub, dcovmat, dim, m, nparms);
    } else if (covfun_name == EXPONENTIAL_SPACETIME) {
        d_exponential_spacetime(covparms, locsub, dcovmat, dim, m, nparms);
    } else if (covfun_name == EXPONENTIAL_SPHERETIME) {
        d_exponential_spheretime(covparms, locsub, dcovmat, dim, m, nparms);
    }
}

#endif