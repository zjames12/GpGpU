#include <chrono>
#include <iostream>


#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <cusolverDn.h>
#include <algorithm>
#include <string>

#include "covmatrix_funs_14.cuh"

// Helper function to check if CUDA functions have executed properly 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

// // Helper function to check if cuBLAS functions have executed properly
// #define cublasErrchk(ans) {cublasAssert((ans), __FILE__, __LINE__);}
// inline void cublasAssert(cublasStatus_t status, const char* file, int line, bool abort = true)
// {
//     if (status != CUBLAS_STATUS_SUCCESS) {
//         printf("GPUassert: %s %s %d\n", "cuBLAS", file, line);
//     }
// }

// // Helper function to check if cuSOLVER functions have executed properly
// #define cusolverErrchk(ans) {cusolverAssert((ans), __FILE__, __LINE__);}
// inline void cusolverAssert(cusolverStatus_t status, const char* file, int line, bool abort = true)
// {
//     if (status != CUSOLVER_STATUS_SUCCESS) {
//         printf("GPUassert: %s %s %d\n", "cuSOLVER", file, line);
//     }
// }


// Computes intermediary quantities for likelihood, gradient, and Fisher information of GP
// using thread-per-observation approach
template <int M, int D>
__global__ void compute_pieces(double* y, double* X, double* NNarray, double* locs,
    double* logdet, double* ySy, double* XSX, double* ySX,
    double* dXSX, double* dySX, double* dySy, double* dlogdet, double* ainfo,
    double* covparms, short covfun_name,
    int n, int p, int m, int dim, int nparms,
    bool profbeta, bool grad_info) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int bsize = std::min(i + 1, m);
    if (i < m || i >= n) {
        return;
    }
    
    double ysub[M];
    double locsub[M * D];
    double X0[M * D];

    // substitute locations
    for (int j = m - 1; j >= 0; j--) {
        ysub[m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
        for (int k = 0; k < dim; k++) {
            locsub[(m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k];
        }
        if (profbeta) {
            for (int k = 0; k < p; k++) {
                X0[(m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
            }
        }
    }
    
    double covmat[M * M];
    for (int j = 0; j < M*M; j++) {
        covmat[j] = 0;
    }
    double temp;
    
    covariance_func(covfun_name, covparms, locsub, covmat, dim, m);
    
    double dcovmat[M * M * D];
    for (int j = 0; j < M*M*D; j++) {
        dcovmat[j] = 0;
    }
    if (grad_info) {
        // calculate derivatives
        d_covariance_func(covfun_name, covparms, locsub, dcovmat, dim, m, nparms);
    }

    // Cholesky decomposition
    //int k, q, j;
    double temp2;
    double diff;

    int r, j, k;
    for (r = 0; r < m + 0; r++) {
        diff = 0;
        for (k = 0; k < r; k++) {
            temp = covmat[r * m + k];
            diff += temp * temp;
        }
        covmat[r * m + r] = sqrt(covmat[r * m + r] - diff);


        for (j = r + 1; j < m + 0; j++) {
            diff = 0;
            for (k = 0; k < r; k++) {
                diff += covmat[r * m + k] * covmat[j * m + k];
            }
            covmat[j * m + r] = (covmat[j * m + r] - diff) / covmat[r * m + r];
        }
    }
    
    // i1 is conditioning set, i2 is response      
    // get last row of cholmat
    
    double choli2[M];
    for (int j = 0; j < M; j++) {
        choli2[j] = 0;
    }
    if (grad_info) {
        //choli2 = backward_solve(cholmat, onevec, m);
        choli2[m - 1] = 1 / covmat[(m - 1) * m + m - 1];

        for (int k = m - 2; k >= 0; k--) {
            double dd = 0.0;
            for (int j = m - 1; j > k; j--) {
                dd += covmat[j * m + k] * choli2[j];
            }
            choli2[k] = (-dd) / covmat[k * m + k];
        }
    }
    
    double LiX0[M * D];
    for (int j = 0; j < M * D; j++) {
        LiX0[j] = 0;
    }
    // do solves with X and y
    for (int j = 0; j < m; j++) {
        for (int k = j + 1; k < m; k++) {
            covmat[j * m + k] = 0.0;
        }
    }
    if (profbeta) {
        // LiX0 = forward_solve_mat(cholmat, X0, m, p);
        for (int k = 0; k < p; k++) {
            LiX0[0 * p + k] = X0[0 * p + k] / covmat[0 * m + 0];
        }

        for (int h = 1; h < m; h++) {
            for (int k = 0; k < p; k++) {
                double dd = 0.0;
                for (int j = 0; j < h; j++) {
                    dd += covmat[h * m + j] * LiX0[j * p + k];
                }
                LiX0[h * p + k] = (X0[h * p + k] - dd) / covmat[h * m + h];
            }
        }
    }
    
    double Liy0[M];
    for (int j = 0; j < M; j++) {
        Liy0[j] = 0;
    }
    Liy0[0] = ysub[0] / covmat[0 * m + 0];

    for (int k = 1; k < m; k++) {
        double dd = 0.0;
        for (int j = 0; j < k; j++) {
            dd += covmat[k * m + j] * Liy0[j];
        }
        Liy0[k] = (ysub[k] - dd) / covmat[k * m + k];

    }
    

    // loglik objects
    logdet[i] = 2.0 * log(covmat[(m - 1) * m + m - 1]);
    
    temp = Liy0[m - 1];
    ySy[i] = temp * temp;
    
    
    if (profbeta) {
        
        temp2 = Liy0[m - 1];
        for (int i1 = 0; i1 < p; i1++) {
            temp = LiX0[(m - 1) * p + i1];
            for (int i2 = 0; i2 <= i1; i2++) {
                
                XSX[i * p * p + i1 * p + i2] = temp * LiX0[(m - 1) * p + i2];
                XSX[i * p * p + i2 * p + i1] = XSX[i * p * p + i1 * p + i2]; 
            }
            ySX[i * p + i1] = temp2 * LiX0[(m - 1) * p + i1];
            
        }
        
    }
    
    double LidSLi3[M];
    double c[M];
    for (int j = 0; j < M; j++) {
        LidSLi3[j] = 0;
        c[0] = 0;
    }
    double LidSLi2[M * D];
    for (int j = 0; j < M * D; j++) {
        LidSLi2[j] = 0;
    }
    double v1[D];
    for (int j = 0; j < D; j++) {
        v1[j] = 0;
    }
    

    if (grad_info) {
        // gradient objects
        // LidSLi3 is last column of Li * (dS_j) * Lit for 1 parameter i
        // LidSLi2 stores these columns in a matrix for all parameters



        for (int j = 0; j < nparms; j++) {
            // compute last column of Li * (dS_j) * Lit
            //arma::vec LidSLi3 = forward_solve(cholmat, dcovmat.slice(j) * choli2);
            // c = dcovmat.slice(j) * choli2
            for (int h = 0; h < m; h++) {
                c[h] = 0;
                temp = 0;
                for (int k = 0; k < m; k++) {
                    temp += dcovmat[h * m * nparms + k * nparms + j] * choli2[k];
                }
                c[h] = temp;
            }

            //LidSLi3 = forward_solve(cholmat, c);      
            LidSLi3[0] = c[0] / covmat[0 * m + 0];

            for (int k = 1; k < m; k++) {
                double dd = 0.0;
                for (int l = 0; l < k; l++) {
                    dd += covmat[k * m + l] * LidSLi3[l];
                }
                LidSLi3[k] = (c[k] - dd) / covmat[k * m + k];
            }

            ////////////////
            //arma::vec v1 = LiX0.t() * LidSLi3;

            for (int h = 0; h < p; h++) {
                v1[h] = 0;
                temp = 0;
                for (int k = 0; k < m; k++) {
                    temp += LiX0[k * p + h] * LidSLi3[k];
                }
                v1[h] = temp;
            }

            ////////////////

            //double s1 = as_scalar(Liy0.t() * LidSLi3);
            double s1 = 0;
            for (int h = 0; h < m; h++) {
                s1 += Liy0[h] * LidSLi3[h];
            }

            ////////////////

            /*(l_dXSX).slice(j) += v1 * LiX0.rows(i2) + (v1 * LiX0.rows(i2)).t() -
                as_scalar(LidSLi3(i2)) * (LiX0.rows(i2).t() * LiX0.rows(i2));*/

            double temp3;
            double temp4 = LidSLi3[m - 1];
            for (int h = 0; h < p; h++) {


                temp = v1[h];
                temp2 = LiX0[(m - 1) * p + h];

                for (int k = 0; k < p; k++) {
                    
                    temp3 = LiX0[(m - 1) * p + k];
                    dXSX[i * p * p * nparms + h * p * nparms + k * nparms + j] = temp * temp3 +
                        (v1[k] - temp4 * temp3) * temp2;
                }
            }
            temp = Liy0[m - 1];
            dySy[i * nparms + j] = (2.0 * s1 - temp4 * temp) * temp;

            /*(l_dySX).col(j) += (s1 * LiX0.rows(i2) + (v1 * Liy0(i2)).t() -
                as_scalar(LidSLi3(i2)) * LiX0.rows(i2) * as_scalar(Liy0(i2))).t();*/

            temp3 = LidSLi3[m - 1];
            for (int h = 0; h < p; h++) {
                temp2 = LiX0[(m - 1) * p + h];
                dySX[i * p * nparms + h * nparms + j] = s1 * temp2 +
                    v1[h] * temp - temp3 * temp2 * temp;
            }

            //(l_dlogdet)(j) += as_scalar(LidSLi3(i2));
            dlogdet[i * nparms + j] = temp3;

            //LidSLi2.col(j) = LidSLi3;
            for (int h = 0; h < m; h++) {
                LidSLi2[h * nparms + j] = LidSLi3[h];
            }
            
        }
        
        // fisher information object
        // bottom right corner gets double counted, so subtract it off
        for (int h = 0; h < nparms; h++) {
            // temp2 = LidSLi2[i * m * nparms + (m - 1) * nparms + h];
            temp2 = LidSLi2[(m - 1) * nparms + h];
            for (int j = 0; j < h + 1; j++) {
                double s = 0;
                for (int l = 0; l < m; l++) {
                    s += LidSLi2[l * nparms + h] * LidSLi2[l * nparms + j];
                }
                ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[(m - 1) * nparms + j] * temp2;
            }
        }
        
    }
    
    
   
}

// Computes intermediary quantities for likelihood, gradient, and Fisher information of GP
// using block-per-observation approach
template <int M, int D, int NPARMS>
__global__ void compute_pieces_blocks(double* y, double* X, double* NNarray, double* locs,
    double* logdet, double* ySy, double* XSX, double* ySX,
    double* dXSX, double* dySX, double* dySy, double* dlogdet, double* ainfo,
    double* covparms, short covfun_name,
    int n, int p, int m, int dim, int nparms,
    bool profbeta, bool grad_info) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = threadIdx.y;
    
    // int l = threadIdx.z;

    if (i < m || i >= n || j >= m || k >= m) {
        return;
    }
    
    __shared__ double ysub[M];
    __shared__ double locsub[M * D];
    __shared__ double X0[M * D];
    
    // substitute locations
    if (j < m && k == 0) {
        ysub[m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
    }
    if (j < m && k < dim) {
        locsub[(m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k];
    }
    if (j < m && k < p && profbeta) {
        X0[(m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
    }
    __syncthreads();
    // calculate covariance
    __shared__ double covmat[M * M];

    
    
    double temp = 0;
    double d = 0;

    if (j < m && k < m){
        
        if (j == k) {
            covmat[j * m + k] = covparms[0] * (1 + covparms[2]);
        } else {
            for (int s = 0; s < dim; s++) {
                temp = (locsub[j * dim + s] - locsub[k * dim + s]) / covparms[1];
                d += temp * temp;
            }
            d = sqrt(d);
            covmat[j * m + k] = covparms[0] * exp(-d);
        }
    }
    __syncthreads();
    
    __shared__ double dcovmat[M * M * NPARMS];
    if (grad_info) {
        dcovmat[j * m * nparms + k * nparms + 0] = 0;
        dcovmat[j * m * nparms + k * nparms + 1] = 0;
        dcovmat[j * m * nparms + k * nparms + 2] = 0;
        double d = 0.0;
        double a = 0;
        for (int r = 0; r < dim; r++) {
            a = (locsub[j * dim + r] - locsub[k * dim + r]) / covparms[1];
            d += a * a;
        }
        d = sqrt(d);
        temp = exp(-d);

        dcovmat[j * m * nparms + k * nparms + 0] += temp;
        dcovmat[j * m * nparms + k * nparms + 1] += covparms[0] * temp * d / covparms[1];
        if (j == k) { // update diagonal entry
            dcovmat[j * m * nparms + k * nparms + 0] += covparms[2];
            dcovmat[j * m * nparms + k * nparms + 2] = covparms[0];
        }
    }
    __syncthreads();
    // cholesky
    if (j == 0 && k == 0) {
        double temp2;
        double diff;

        int r, s, t;
        for (r = 0; r < m + 0; r++) {
            diff = 0;
            for (t = 0; t < r; t++) {
                temp = covmat[r * m + t];
                diff += temp * temp;
            }
            covmat[r * m + r] = sqrt(covmat[r * m + r] - diff);
            for (s = r + 1; s < m + 0; s++) {
                diff = 0;
                for (t = 0; t < r; t++) {
                    diff += covmat[r * m + t] * covmat[s * m + t];
                }
                covmat[s * m + r] = (covmat[s * m + r] - diff) / covmat[r * m + r];
            }
        }
    }

    double choli2[M];
    if (grad_info && j == 0 && k == 0) {
        for (int r = 0; r < M; r++) {
            choli2[r] = 0;
        }
        
        choli2[m - 1] = 1 / covmat[(m - 1) * m + m - 1];

        for (int k = m - 2; k >= 0; k--) {
            double dd = 0.0;
            for (int r = m - 1; r > k; r--) {
                dd += covmat[r * m + k] * choli2[r];
            }
            choli2[k] = (-dd) / covmat[k * m + k];
        }
    }

    double LiX0[M * D];

    if (profbeta) {
        if (j == 0 && k == 0) {
            for (int r = 0; r < M * D; r++) {
                LiX0[r] = 0;
            }
            
            for (int r = 0; r < p; r++) {
                LiX0[0 * p + r] = X0[0 * p + r] / covmat[0 * m + 0];
            }

            for (int h = 1; h < m; h++) {
                for (int r = 0; r < p; r++) {
                    double dd = 0.0;
                    for (int j = 0; j < h; j++) {
                        dd += covmat[h * m + j] * LiX0[j * p + r];
                    }
                    LiX0[h * p + r] = (X0[h * p + r] - dd) / covmat[h * m + h];
                }
            }
        }
    }
    // __syncthreads();
    
    
    double Liy0[M];
    if (j == 0 && k == 0) {
        for (int r = 0; r < M; r++) {
            Liy0[r] = 0;
        }
        
        Liy0[0] = ysub[0] / covmat[0 * m + 0];
        for (int r = 1; r < m; r++) {
            double dd = 0.0;
            for (int s = 0; s < r; s++) {
                dd += covmat[r * m + s] * Liy0[s];
            }
            Liy0[r] = (ysub[r] - dd) / covmat[r * m + r];
        }

        logdet[i] = 2.0 * log(covmat[(m - 1) * m + m - 1]);
        temp = Liy0[m - 1];
        ySy[i] = temp * temp;

       
    }

    if (profbeta) {
        if (j == 0 && k == 0) {
            double temp2 = Liy0[m - 1];
            for (int i1 = 0; i1 < p; i1++) {
                temp = LiX0[(m - 1) * p + i1];
                for (int i2 = 0; i2 <= i1; i2++) {
                    XSX[i * p * p + i1 * p + i2] = temp * LiX0[(m - 1) * p + i2];
                    XSX[i * p * p + i2 * p + i1] = XSX[i * p * p + i1 * p + i2]; 
                }
                ySX[i * p + i1] = temp2 * LiX0[(m - 1) * p + i1];
            }
        }
    }

    
    double temp2 = 0;
    if (grad_info && j == 0 && k == 0) {
        double LidSLi3[M];
        double c[M];
        for (int r = 0; r < M; r++) {
            LidSLi3[r] = 0;
            c[0] = 0;
        }
        double LidSLi2[M * D];
        for (int r = 0; r < M * D; r++) {
            LidSLi2[r] = 0;
        }
        double v1[D];
        for (int r = 0; r < D; r++) {
            v1[r] = 0;
        }
        
        for (int r = 0; r < nparms; r++) {
            // compute last column of Li * (dS_j) * Lit
            //arma::vec LidSLi3 = forward_solve(cholmat, dcovmat.slice(j) * choli2);
            for (int h = 0; h < m; h++) {
                c[h] = 0;
                temp = 0;
                for (int k = 0; k < m; k++) {
                    temp += dcovmat[h * m * nparms + k * nparms + r] * choli2[k];
                }
                c[h] = temp;
            }

            //LidSLi3 = forward_solve(cholmat, c);      
            LidSLi3[0] = c[0] / covmat[0 * m + 0];

            for (int k = 1; k < m; k++) {
                double dd = 0.0;
                for (int l = 0; l < k; l++) {
                    dd += covmat[k * m + l] * LidSLi3[l];
                }
                LidSLi3[k] = (c[k] - dd) / covmat[k * m + k];
            }

            ////////////////

            for (int h = 0; h < p; h++) {
                v1[h] = 0;
                temp = 0;
                for (int k = 0; k < m; k++) {
                    temp += LiX0[k * p + h] * LidSLi3[k];
                }
                v1[h] = temp;
            }

            ////////////////

            double s1 = 0;
            for (int h = 0; h < m; h++) {
                s1 += Liy0[h] * LidSLi3[h];
            }

            ////////////////
            double temp3;
            
            double temp4 = LidSLi3[m - 1];
            for (int h = 0; h < p; h++) {
                temp = v1[h];
                temp2 = LiX0[(m - 1) * p + h];

                for (int k = 0; k < p; k++) {
                    temp3 = LiX0[(m - 1) * p + k];
                    dXSX[i * p * p * nparms + h * p * nparms + k * nparms + r] = temp * temp3 +
                        (v1[k] - temp4 * temp3) * temp2;
                }
            }
            temp = Liy0[m - 1];
            ///////////////

            dySy[i * nparms + r] = (2.0 * s1 - temp4 * temp) * temp;
            temp3 = LidSLi3[m - 1];
            for (int h = 0; h < p; h++) {
                temp2 = LiX0[(m - 1) * p + h];
                dySX[i * p * nparms + h * nparms + r] = s1 * temp2 +
                    v1[h] * temp - temp3 * temp2 * temp;
            }

            dlogdet[i * nparms + r] = temp3;

            for (int h = 0; h < m; h++) {
                LidSLi2[h * nparms + r] = LidSLi3[h];
            }

        }
        
        // fisher information object
        // bottom right corner gets double counted, so subtract it off
        for (int h = 0; h < nparms; h++) {
            temp2 = LidSLi2[(m - 1) * nparms + h];
            for (int r = 0; r < h + 1; r++) {
                double s = 0;
                for (int l = 0; l < m; l++) {
                    s += LidSLi2[l * nparms + h] * LidSLi2[l * nparms + r];
                }
                ainfo[i * nparms * nparms + h * nparms + r] = s - 0.5 * LidSLi2[(m - 1) * nparms + r] * temp2;
            }
        }
    }
}

// Function called by C++ code to run kernels
extern "C"
void call_compute_pieces_gpu(
    const double* covparms,
    const short covfun_name,
    const double* locs,
    const double* NNarray,
    const double* y,
    const double* X,
    double* XSX,
    double* ySX,
    double* ySy,
    double* logdet,
    double* dXSX,
    double* dySX,
    double* dySy,
    double* dlogdet,
    double* ainfo,
    const bool profbeta,
    const bool grad_info,
    const int n,
    const int m,
    const int p,
    const int nparms,
    const int dim
) {
    double* d_locs;
    double* d_NNarray;
    double* d_y;
    double* d_X;
    double* d_covparms;
    
    gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
    gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_y, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_X, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_covparms, sizeof(double) * nparms));


    double* d_ySX;
    double* d_XSX;
    double* d_ySy;
    double* d_logdet;
    double* d_dXSX;
    double* d_dySX;
    double* d_dySy;
    double* d_dlogdet;
    double* d_ainfo;

	gpuErrchk(cudaMalloc((void**)&d_ySX, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_XSX, sizeof(double) * n * p * p));
    gpuErrchk(cudaMalloc((void**)&d_ySy, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_logdet, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_dXSX, sizeof(double) * n * p * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySX, sizeof(double) * n * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySy, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dlogdet, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_ainfo, sizeof(double) * n * nparms * nparms));

    gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_X, X, sizeof(double) * n * p, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_covparms, covparms, sizeof(double) * nparms, cudaMemcpyHostToDevice));
    
    int block_size = 64;
    int grid_size = ((n + block_size) / block_size);
    
    if (dim <= 4) {
        if (m <= 31) {
            compute_pieces<31, 4><<<grid_size, block_size>>> (d_y, d_X, d_NNarray, d_locs,
                d_logdet, d_ySy, d_XSX, d_ySX,
                d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
                d_covparms, covfun_name,
                n, p, m, dim, nparms,
                profbeta, grad_info);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        } else if (m <= 61) {
            compute_pieces<61, 4><<<grid_size, block_size>>> (d_y, d_X, d_NNarray, d_locs,
                d_logdet, d_ySy, d_XSX, d_ySX,
                d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
                d_covparms, covfun_name,
                n, p, m, dim, nparms,
                profbeta, grad_info);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        } else if (m <= 91) {
            compute_pieces<91, 4><<<grid_size, block_size>>> (d_y, d_X, d_NNarray, d_locs,
                d_logdet, d_ySy, d_XSX, d_ySX,
                d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
                d_covparms, covfun_name,
                n, p, m, dim, nparms,
                profbeta, grad_info);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
    }

    // int grid_size = n;
    // dim3 block_size = dim3(m, m);

    // if (dim <= 4) {
    //     if (m <= 31) {
    //         compute_pieces_blocks<31, 4, 3><<<grid_size, dim3(m, m)>>> (d_y, d_X, d_NNarray, d_locs,
    //             d_logdet, d_ySy, d_XSX, d_ySX,
    //             d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
    //             d_covparms, covfun_name,
    //             n, p, m, dim, nparms,
    //             profbeta, grad_info);
    //         gpuErrchk(cudaPeekAtLastError());
    //         gpuErrchk(cudaDeviceSynchronize());
    //     } else if (m <= 61) {
    //         // compute_pieces_blocks<61, 3, 3><<<grid_size, dim3(m, m)>>> (d_y, d_X, d_NNarray, d_locs,
    //         //     d_logdet, d_ySy, d_XSX, d_ySX,
    //         //     d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
    //         //     d_covparms, covfun_name,
    //         //     n, p, m, dim, nparms,
    //         //     profbeta, grad_info);
    //         // gpuErrchk(cudaPeekAtLastError());
    //         // gpuErrchk(cudaDeviceSynchronize());
    //     } else if (m <= 91) {
    //         // compute_pieces_blocks<91, 4><<<grid_size, dim3(m, m)>>> (d_y, d_X, d_NNarray, d_locs,
    //         //     d_logdet, d_ySy, d_XSX, d_ySX,
    //         //     d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
    //         //     d_covparms, covfun_name,
    //         //     n, p, m, dim, nparms,
    //         //     profbeta, grad_info);
    //         // gpuErrchk(cudaPeekAtLastError());
    //         // gpuErrchk(cudaDeviceSynchronize());
    //     }
    // }

    double* l_ySy = (double*)malloc(sizeof(double) * n);
    double* l_logdet = (double*)malloc(sizeof(double) * n);
    double* l_ySX = (double*)malloc(sizeof(double) * n * p);
    double* l_XSX = (double*)malloc(sizeof(double) * n * p * p);
    double* l_dySX = (double*)malloc(sizeof(double) * n * p * nparms);
    double* l_dXSX = (double*)malloc(sizeof(double) * n * p * p * nparms);
    double* l_dySy = (double*)malloc(sizeof(double) * n * nparms);
    double* l_dlogdet = (double*)malloc(sizeof(double) * n * nparms);
    double* l_ainfo = (double*)malloc(sizeof(double) * n * nparms * nparms);
    
    gpuErrchk(cudaMemcpy(l_ySy, d_ySy, sizeof(double) * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_logdet, d_logdet, sizeof(double) * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_ySX, d_ySX, sizeof(double) * n * p, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_XSX, d_XSX, sizeof(double) * n * p * p, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_dySX, d_dySX, sizeof(double) * n * p * nparms, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_dXSX, d_dXSX, sizeof(double) * n * p * p * nparms, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_dySy, d_dySy, sizeof(double) * n * nparms, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_dlogdet, d_dlogdet, sizeof(double) * n * nparms, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_ainfo, d_ainfo, sizeof(double) * n * nparms * nparms, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_ySX));
    gpuErrchk(cudaFree(d_XSX));
    gpuErrchk(cudaFree(d_ySy));
    gpuErrchk(cudaFree(d_logdet));
    gpuErrchk(cudaFree(d_dXSX));
    gpuErrchk(cudaFree(d_dySX));
    gpuErrchk(cudaFree(d_dySy));
    gpuErrchk(cudaFree(d_dlogdet));
    gpuErrchk(cudaFree(d_ainfo));

    gpuErrchk(cudaFree(d_locs));
    gpuErrchk(cudaFree(d_NNarray));
    gpuErrchk(cudaFree(d_y));
    gpuErrchk(cudaFree(d_X));

    ySy[0] = 0;
    logdet[0] = 0;
    for (int i = m; i < n; i++) {
        // printf("%i, %f\n", i, l_ySy[i]);
        ySy[0] += l_ySy[i];
        logdet[0] += l_logdet[i];
        
        for (int j = 0; j < p; j++) {
            if (i == m) {
                ySX[j] = 0;
            }
            // printf("(%i, %f)", i, l_ySX[i * p + j]);
            ySX[j] += l_ySX[i * p + j];
            for (int k = 0; k < p; k++) {
                if (i == m) {
                    XSX[j * p + k] = 0;
                }
                XSX[j * p + k] += l_XSX[i * p * p + j * p + k];
                for (int l = 0; l < nparms; l++) {
                    if (i == m) {
                        dXSX[j * p * nparms + k * nparms + l] = 0;
                    }
                    dXSX[j * p * nparms + k * nparms + l] += l_dXSX[i * p * p * nparms + j * p * nparms + k * nparms + l];
                    //dXSX[j * p * nparms + k * nparms + l] += 0;
                }
            }
            for (int k = 0; k < nparms; k++) {
                if (i == m) {
                    dySX[j * nparms + k] = 0;
                }
                dySX[j * nparms + k] += l_dySX[i * p * nparms + j * nparms + k];
                //printf("%f ", l_dySX[i * p * nparms + j * nparms + k]);
            }
        }
        for (int j = 0; j < nparms; j++) {
            if (i == m) {
                dySy[j] = 0;
                dlogdet[j] = 0;
            }
            dySy[j] += l_dySy[i * nparms + j];
            dlogdet[j] += l_dlogdet[i * nparms + j];
            for (int k = 0; k < nparms; k++) {
                if (i == m) {
                    ainfo[j * nparms + k] = 0;
                }
                ainfo[j * nparms + k] += l_ainfo[i * nparms * nparms + j * nparms + k];
            }
        }
    }
    
    free(l_ySy);
    free(l_logdet);
    free(l_ySX);
    free(l_XSX);
    free(l_dySX);
    free(l_dXSX);
    free(l_dySy);
    free(l_dlogdet);
    free(l_ainfo);
    
}

__global__ void substitute_batched_kernel_cublas(double* NNarray, double* locs, double* sub_locs[],
    int n, int m, int dim) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = threadIdx.y;

    if (i < m) {
        return;
    }
    sub_locs[i][(m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k];
}
__global__ void substitute_X0_batched_kernel_cublas(double* NNarray, double* X, double* X0[],
    int n, int m, int dim, int p) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = threadIdx.y;

    if (i < m || j >= m || k >= p) {
        return;
    }
    X0[i][(m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
    
}
__global__ void substitute_ysub_batched_kernel_cublas(double* NNarray, double* y, double* ysub[],
    int n, int m, int dim) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i < m || j >= m) {
        return;
    }
    ysub[i][m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
}

__global__ void covariance_batched_kernel_cublas(double* sub_locs[], double* cov[], double* covparms, int n, int m, int dim) {
    int i = blockIdx.x;
    int i1 = threadIdx.x;
    int i2 = threadIdx.y;
    
    if (i < m) {
        return;
    }
    while (i1 < m) {
        while (i2 < m) {
            if (i1 == i2) {
                cov[i][i1 * m + i2] = covparms[0] * (1 + covparms[2]);
            } else {
                double d = 0;
                double temp;
                for (int k = 0; k < dim; k++) {
                    temp = sub_locs[i][i1 * dim + k] / covparms[1] - sub_locs[i][i2 * dim + k] / covparms[1];
                    d += temp * temp;
                }
                d = sqrt(d);
                
                cov[i][i1 * m + i2] = exp(-d) * covparms[0];
                // cov[i][i2 * m + i1] = cov[i][i1 * m + i2];
            }
            i2 += blockDim.y;
        }
        i2 = threadIdx.y;
        i1 += blockDim.x;
    }
}

__global__ void dcovariance_batched_kernel_cublas(double* sub_locs[], double* dcovmat[], double* covparms, int n, int m, int dim, int nparms) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int i1 = threadIdx.y;
    int i2 = threadIdx.z;

    while (i1 < m) {
        while (i2 < m) {
            if (j == 2) {
                // dcovmat[i * nparms + j][i1 * m + i2] = covparms[0];
                if (i1 == i2) {
                    dcovmat[j * n + i][i1 * m + i2] = covparms[0];
                } else {
                    dcovmat[j * n + i][i1 * m + i2] = 0;
                }
            }
            double d = 0.0;
            double a = 0;
            for (int r = 0; r < dim; r++) {
                a = (sub_locs[i][i1 * dim + r] - sub_locs[i][i2 * dim + r]) / covparms[1];
                d += a * a;
            }
            d = sqrt(d);
            double temp = exp(-d);

            if (j == 0) {
                // dcovmat[i * nparms + j][i1 * m + i2] = temp;
                dcovmat[j * n + i][i1 * m + i2] = temp;
                if (i1 == i2) {
                    // dcovmat[i * nparms + j][i1 * m + i2] += covparms[2];
                    dcovmat[j * n + i][i1 * m + i2] += covparms[2];
                }
            } else if (j == 1) {
                // dcovmat[i * nparms + j][i1 * m + i2] = covparms[0] * temp * d / covparms[1];
                dcovmat[j * n + i][i1 * m + i2] = covparms[0] * temp * d / covparms[1];
            }
            i2 += blockDim.z;
        }
        i2 = threadIdx.z;
        i1 += blockDim.y;
    }
}

__global__ void grad_info_batched_kernel(
    double** covmat, double** dcovmat,
    double* dXSX, double* dySX, double* dySy, double* dlogdet, double* ainfo,
    double* LidSLi3, double* LidSLi2, double* v1, double* c,
    double** LiX0, double** Liy0, double** choli2,
    int n, int p, int m, int dim, int nparms
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m || i >= n) {
        return;
    }
    
    
    double temp, temp2;
    for (int j = 0; j < nparms; j++) {
        // compute last column of Li * (dS_j) * Lit
        //arma::vec LidSLi3 = forward_solve(cholmat, dcovmat.slice(j) * choli2);
        // c = dcovmat.slice(j) * choli2
        for (int h = 0; h < m; h++) {
            c[i * m + h] = 0;
            temp = 0;
            for (int k = 0; k < m; k++) {
                temp += dcovmat[j * n + i][h * m + k] * choli2[i][k];
            }
            c[i * m + h] = temp;
        }
        
        //LidSLi3 = forward_solve(cholmat, c);      
        LidSLi3[i * m + 0] = c[i * m + 0] / covmat[i][0 * m + 0];

        for (int k = 1; k < m; k++) {
            double dd = 0.0;
            for (int l = 0; l < k; l++) {
                dd += covmat[i][l * m + k] * LidSLi3[i * m + l];
            }
            LidSLi3[i * m + k] = (c[i * m + k] - dd) / covmat[i][k * m + k];
        }
        //arma::vec v1 = LiX0.t() * LidSLi3;

        for (int h = 0; h < p; h++) {
            v1[i * p + h] = 0;
            temp = 0;
            for (int k = 0; k < m; k++) {
                temp += LiX0[i][k * p + h] * LidSLi3[i * m + k];
            }
            v1[i * p + h] = temp;
        }
        
        //double s1 = as_scalar(Liy0.t() * LidSLi3);
        double s1 = 0;
        for (int h = 0; h < m; h++) {
            // s1 += Liy0[i * m + h] * LidSLi3[i * m + h];
            s1 += Liy0[i][h] * LidSLi3[i * m + h];
        }

        //(l_dXSX).slice(j) += v1 * LiX0.rows(i2) + (v1 * LiX0.rows(i2)).t() -
          //  as_scalar(LidSLi3(i2)) * (LiX0.rows(i2).t() * LiX0.rows(i2));

        double temp3;
        double temp4 = LidSLi3[i * m + m - 1];
        for (int h = 0; h < p; h++) {
            temp = v1[i * p + h];
            temp2 = LiX0[i][(m - 1) * p + h];

            for (int k = 0; k < p; k++) {
                temp3 = LiX0[i][(m - 1) * p + k];
                dXSX[i * p * p * nparms + h * p * nparms + k * nparms + j] = temp * temp3 +
                    (v1[i * p + k] - temp4 * temp3) * temp2;
                    
            }
        }
        temp = Liy0[i][m - 1];
        dySy[i * nparms + j] = (2.0 * s1 - temp4 * temp) * temp;

        //(l_dySX).col(j) += (s1 * LiX0.rows(i2) + (v1 * Liy0(i2)).t() -
         //   as_scalar(LidSLi3(i2)) * LiX0.rows(i2) * as_scalar(Liy0(i2))).t();

        temp3 = LidSLi3[i * m + m - 1];
        for (int h = 0; h < p; h++) {
            temp2 = LiX0[i][(m - 1) * p + h];
            dySX[i * p * nparms + h * nparms + j] = s1 * temp2 +
                v1[i * p + h] * temp - temp3 * temp2 * temp;
        }

        //(l_dlogdet)(j) += as_scalar(LidSLi3(i2));
        dlogdet[i * nparms + j] = temp3;

        //LidSLi2.col(j) = LidSLi3;
        for (int h = 0; h < m; h++) {
            LidSLi2[i * m * nparms + h * nparms + j] = LidSLi3[i * m + h];
        }
    }
    
    // fisher information object
    // bottom right corner gets double counted, so subtract it off
    for (int h = 0; h < nparms; h++) {
        temp2 = LidSLi2[i * m * nparms + (m - 1) * nparms + h];
        // temp2 = LidSLi2[(m - 1) * nparms + h];
        for (int j = 0; j < h + 1; j++) {
            double s = 0;
            for (int l = 0; l < m; l++) {
                s += LidSLi2[i * m * nparms + l * nparms + h] * LidSLi2[i * m * nparms + l * nparms + j];
            }
            ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[i * m * nparms + (m - 1) * nparms + j] * temp2;
        }
    }
}

// extern "C"
// void call_compute_pieces_gpu_batched(
//     const double* covparms,
//     const short covfun_name,
//     const double* locs,
//     const double* NNarray,
//     const double* y,
//     const double* X,
//     double* XSX,
//     double* ySX,
//     double* ySy,
//     double* logdet,
//     double* dXSX,
//     double* dySX,
//     double* dySy,
//     double* dlogdet,
//     double* ainfo,
//     const bool profbeta,
//     const bool grad_info,
//     const int n,
//     const int m,
//     const int p,
//     const int nparms,
//     const int dim
// ) {
//     //transfer to gpu
//     double* d_covparms, * d_locs, * d_NNarray, *d_y, *d_X;
//     gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
//     gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
//     gpuErrchk(cudaMalloc((void**)&d_covparms, sizeof(double) * nparms));
//     gpuErrchk(cudaMalloc((void**)&d_y, sizeof(double) * n));
//     gpuErrchk(cudaMalloc((void**)&d_X, sizeof(double) * n * p));


//     gpuErrchk(cudaMemcpy(d_covparms, covparms, sizeof(double) *  nparms, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_X, X, sizeof(double) * n * p, cudaMemcpyHostToDevice));


//     double** sublocs = (double**)malloc(sizeof(double*) * n);
//     double** d_sublocs = NULL;
//     double** cov = (double**)malloc(sizeof(double*) * n);
//     double** d_cov = NULL;
//     double** ysub = (double**)malloc(sizeof(double*) * n);
//     double** d_ysub = NULL;
//     double** X0 = (double**)malloc(sizeof(double*) * n);
//     double** d_X0 = NULL;

//     cudaMalloc((void**)&sublocs[0], sizeof(double) * n * m * dim);
//     cudaMalloc((void**)&cov[0], sizeof(double) * n * m * m);
//     cudaMalloc((void**)&ysub[0], sizeof(double) * n * m);
//     cudaMalloc((void**)&X0[0], sizeof(double) * n * m * p);
//     for (int j = 1; j < n; j++) {
//         sublocs[j] = sublocs[j-1] + m * dim;
//         cov[j] = cov[j-1] + m * m;
//         ysub[j] = ysub[j-1] + m;
//         X0[j] = X0[j-1] + m * p;
//     }
//     cudaMalloc((void**)&d_sublocs, sizeof(double*) * n);
//     cudaMemcpy(d_sublocs, sublocs, sizeof(double*) * n, cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_cov, sizeof(double*) * n);
//     cudaMemcpy(d_cov, cov, sizeof(double*) * n, cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_ysub, sizeof(double*) * n);
//     cudaMemcpy(d_ysub, ysub, sizeof(double*) * n, cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_X0, sizeof(double*) * n);
//     cudaMemcpy(d_X0, X0, sizeof(double*) * n, cudaMemcpyHostToDevice);
    
    
//     double** dcovmat = (double**)malloc(sizeof(double*) * n * nparms);
//     double** d_dcovmat = NULL;
//     cudaMalloc((void**)&dcovmat[0], sizeof(double) * n * nparms * m * m);
//     for (int j = 1; j < n * nparms; j++) {
//         dcovmat[j] = dcovmat[j-1] + m * m;
//     }
//     cudaMalloc((void**)&d_dcovmat, sizeof(double*) * n * nparms);
//     cudaMemcpy(d_dcovmat, dcovmat, sizeof(double*) * n * nparms, cudaMemcpyHostToDevice);

//     // onevec
//     double** onevec = (double**)malloc(sizeof(double*) * n);
//     double** d_onevec = NULL;
//     cudaMalloc((void**)&onevec[0], sizeof(double) * n * m);

//     double* lonevec = (double*)calloc(n * m, sizeof(double));
//     for (int j = 0; j < n; j++) {
//         lonevec[j * m + (m-1)] = 1.0;
//     }
//     cudaMemcpy(onevec[0], lonevec, sizeof(double) * n * m, cudaMemcpyHostToDevice);
//     for (int j = 1; j < n; j++) {
//         onevec[j] = onevec[j-1] + m;
//     }
//     cudaMalloc((void**)&d_onevec, sizeof(double*) * n * m);
//     cudaMemcpy(d_onevec, onevec, sizeof(double*) * n * m, cudaMemcpyHostToDevice);
    
//     substitute_batched_kernel_cublas << <dim3(n, 1, 1), dim3(m, dim) >> > (d_NNarray, d_locs, d_sublocs, n, m, dim);
//     gpuErrchk(cudaPeekAtLastError());
//     // gpuErrchk(cudaDeviceSynchronize());

//     substitute_X0_batched_kernel_cublas << <dim3(n, 1, 1), dim3(m, p) >> > (d_NNarray, d_X, d_X0, n, m, dim, p);
//     gpuErrchk(cudaPeekAtLastError());

//     substitute_ysub_batched_kernel_cublas << <dim3(n, 1, 1), m >> > (d_NNarray, d_y, d_ysub, n, m, dim);
//     gpuErrchk(cudaPeekAtLastError());
//     gpuErrchk(cudaDeviceSynchronize());

//     covariance_batched_kernel_cublas << <dim3(n,1,1), dim3(31, 31) >> > (d_sublocs, d_cov, d_covparms, n, m, dim);
//     gpuErrchk(cudaPeekAtLastError());

//     if (grad_info) {
//         dcovariance_batched_kernel_cublas << <dim3(n,1,1), dim3(nparms, 18, 18) >> > (d_sublocs, d_dcovmat, d_covparms, n, m, dim, nparms);
//         gpuErrchk(cudaPeekAtLastError());
//     }

//     gpuErrchk(cudaDeviceSynchronize());

//     cusolverDnHandle_t cusolver_handle;
//     cusolverDnCreate(&cusolver_handle);
//     int* d_info;
//     gpuErrchk(cudaMalloc((void**)&d_info, n * sizeof(int)));

//     cusolverErrchk(cusolverDnDpotrfBatched(cusolver_handle, CUBLAS_FILL_MODE_LOWER, m, d_cov, m, d_info, n));
//     gpuErrchk(cudaDeviceSynchronize());

//     cusolverErrchk(cusolverDnDestroy(cusolver_handle));
    
//     cublasHandle_t cublas_handle;
//     cublasCreate_v2(&cublas_handle);

//     // choli2 = backward_solve( cholmat, onevec );
//     if (grad_info) {
//         const double alpha = 1.f;
//         cublasErrchk(cublasDtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, 1, &alpha, d_cov, m, d_onevec, m, n));
//         gpuErrchk(cudaDeviceSynchronize());
//     }

//     // LiX0 = forward_solve_mat( cholmat, X0 );
//     double* LiX0 = (double*)malloc(n * m * p * sizeof(double));
//     if (profbeta) {
//         const double alpha = 1.f;
//         cublasErrchk(cublasDtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, p, &alpha, d_cov, m, d_X0, m, n));
//         gpuErrchk(cudaDeviceSynchronize());

//         gpuErrchk(cudaMemcpy(LiX0, X0[0], sizeof(double) * n * m * p, cudaMemcpyDeviceToHost));
//     }
    
//     const double alpha = 1.f;
//     cublasErrchk(cublasDtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, 1, &alpha, d_cov, m, d_ysub, m, n));
//     gpuErrchk(cudaDeviceSynchronize());

//     double* lcov = (double*)malloc(sizeof(double) * n * m * m);
//     double* Liy0 = (double*)malloc(sizeof(double) * n * m);
//     gpuErrchk(cudaMemcpy(lcov, cov[0], sizeof(double) * n * m * m, cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaMemcpy(Liy0, ysub[0], sizeof(double) * n * m, cudaMemcpyDeviceToHost));

//     cublasErrchk(cublasDestroy(cublas_handle));

//     logdet[0] = 0.0;
//     ySy[0] = 0.0;
//     for (int i = m; i < n; i++) {
//         logdet[0] += 2.0 * log(lcov[i * m * m + (m - 1) * m + m - 1]);
//         ySy[0] += Liy0[i * m + m - 1] * Liy0[i * m + m - 1];
//     }

//     if (profbeta) {
//         for (int i = m; i < n; i++) {
//             for (int i1 = 0; i1 < p; i1++) {
//                 for (int i2 = 0; i2 < p; i2++) {
//                     if (i == m) {
//                         XSX[i1 * p + i2] = 0;
//                     }
//                     XSX[i1 * p + i2] += LiX0[i * m * p + (m - 1) * p + i1] * LiX0[i * m * p + (m - 1) * p + i2];
//                 }
//                 if (i == m) {
//                     ySX[i1] = 0;
//                 }
//                 ySX[i1] += Liy0[i * m + m - 1] * LiX0[i * m * p + (m - 1) * p + i1];
//             }
//         }
//     }

//     if (grad_info) {
        
//         double* d_dXSX;
//         double* d_dySX;
//         double* d_dySy;
//         double* d_dlogdet;
//         double* d_ainfo;
//         gpuErrchk(cudaMalloc((void**)&d_dXSX, sizeof(double) * n * p * p * nparms));
//         gpuErrchk(cudaMalloc((void**)&d_dySX, sizeof(double) * n * p * nparms));
//         gpuErrchk(cudaMalloc((void**)&d_dySy, sizeof(double) * n * nparms));
//         gpuErrchk(cudaMalloc((void**)&d_dlogdet, sizeof(double) * n * nparms));
//         gpuErrchk(cudaMalloc((void**)&d_ainfo, sizeof(double) * n * nparms * nparms));

//         double* d_LidSLi3;
//         double* d_LidSLi2;
//         double* d_v1;
//         double* d_c;
        
//         gpuErrchk(cudaMalloc((void**)&d_LidSLi3, sizeof(double) * n * m));
//         gpuErrchk(cudaMalloc((void**)&d_LidSLi2, sizeof(double) * n * m * nparms));
//         gpuErrchk(cudaMalloc((void**)&d_v1, sizeof(double) * n * p));
//         gpuErrchk(cudaMalloc((void**)&d_c, sizeof(double) * n * m));
        
//         int block_size = 64;
//         int grid_size = ((n + block_size) / block_size);
//         grad_info_batched_kernel<<<grid_size, block_size>>>(
//             d_cov, d_dcovmat,
//             d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
//             d_LidSLi3, d_LidSLi2, d_v1, d_c,
//             d_X0, d_ysub, d_onevec,
//             n, p, m, dim, nparms
//         );
//         gpuErrchk(cudaPeekAtLastError());
//         gpuErrchk(cudaDeviceSynchronize());
        
//         double* l_dySX = (double*)malloc(sizeof(double) * n * p * nparms);
//         double* l_dXSX = (double*)malloc(sizeof(double) * n * p * p * nparms);
//         double* l_dySy = (double*)malloc(sizeof(double) * n * nparms);
//         double* l_dlogdet = (double*)malloc(sizeof(double) * n * nparms);
//         double* l_ainfo = (double*)malloc(sizeof(double) * n * nparms * nparms);
        
//         gpuErrchk(cudaMemcpy(l_dySX, d_dySX, sizeof(double) * n * p * nparms, cudaMemcpyDeviceToHost));
//         gpuErrchk(cudaMemcpy(l_dXSX, d_dXSX, sizeof(double) * n * p * p * nparms, cudaMemcpyDeviceToHost));
//         gpuErrchk(cudaMemcpy(l_dySy, d_dySy, sizeof(double) * n * nparms, cudaMemcpyDeviceToHost));
//         gpuErrchk(cudaMemcpy(l_dlogdet, d_dlogdet, sizeof(double) * n * nparms, cudaMemcpyDeviceToHost));
//         gpuErrchk(cudaMemcpy(l_ainfo, d_ainfo, sizeof(double) * n * nparms * nparms, cudaMemcpyDeviceToHost));

//         gpuErrchk(cudaFree(d_locs));
//         gpuErrchk(cudaFree(d_NNarray));
//         gpuErrchk(cudaFree(d_covparms));
//         gpuErrchk(cudaFree(d_y));
//         gpuErrchk(cudaFree(d_X));

//         gpuErrchk(cudaFree(sublocs[0]));
//         gpuErrchk(cudaFree(cov[0]));
//         gpuErrchk(cudaFree(ysub[0]));
//         gpuErrchk(cudaFree(X0[0]));

//         gpuErrchk(cudaFree(d_sublocs));
//         gpuErrchk(cudaFree(d_cov));
//         gpuErrchk(cudaFree(d_ysub));
//         gpuErrchk(cudaFree(d_X0));

//         gpuErrchk(cudaFree(dcovmat[0]));
//         gpuErrchk(cudaFree(d_dcovmat));
        
//         gpuErrchk(cudaFree(onevec[0]));
//         gpuErrchk(cudaFree(d_onevec));

//         gpuErrchk(cudaFree(d_dXSX));
//         gpuErrchk(cudaFree(d_dySX));
//         gpuErrchk(cudaFree(d_dySy));
//         gpuErrchk(cudaFree(d_dlogdet));
//         gpuErrchk(cudaFree(d_ainfo));

//         gpuErrchk(cudaFree(d_LidSLi3));
//         gpuErrchk(cudaFree(d_LidSLi2));
//         gpuErrchk(cudaFree(d_v1));
//         gpuErrchk(cudaFree(d_c));
        
//         for (int i = m; i < n; i++) {
//             for (int j = 0; j < p; j++) {
//                 for (int k = 0; k < p; k++) {
//                     for (int l = 0; l < nparms; l++) {
//                         if (i == m) {
//                             dXSX[j * p * nparms + k * nparms + l] = 0;
//                         }
//                         dXSX[j * p * nparms + k * nparms + l] += l_dXSX[i * p * p * nparms + j * p * nparms + k * nparms + l];
//                     }
//                 }
//                 for (int k = 0; k < nparms; k++) {
//                     if (i == m) {
//                         dySX[j * nparms + k] = 0;
//                     }
//                     dySX[j * nparms + k] += l_dySX[i * p * nparms + j * nparms + k];
//                 }
//             }
//             for (int j = 0; j < nparms; j++) {
//                 if (i == m) {
//                     dySy[j] = 0;
//                     dlogdet[j] = 0;
//                 }
//                 dySy[j] += l_dySy[i * nparms + j];
//                 dlogdet[j] += l_dlogdet[i * nparms + j];
//                 for (int k = 0; k < nparms; k++) {
//                     if (i == m) {
//                         ainfo[j * nparms + k] = 0;
//                     }
//                     ainfo[j * nparms + k] += l_ainfo[i * nparms * nparms + j * nparms + k];
//                 }
//             }
//         }

//         free(l_dySX);
//         free(l_dXSX);
//         free(l_dySy);
//         free(l_dlogdet);
//         free(l_ainfo);
//     }

//     free(sublocs);
//     free(cov);
//     free(ysub);
//     free(X0);
//     free(dcovmat);
//     free(lonevec);
//     free(LiX0);
//     free(lcov);
//     free(Liy0);
// }