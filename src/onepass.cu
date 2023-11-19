

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
//' GPU Error check function
//`
//' Kernels do not throw exceptions. They instead return exit codes. If the exit code is
//` not \code{cudaSuccess} an error message is printed and the code is aborted.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    /*printf(cudaGetErrorString(code));
    printf("\n");*/
    if (code != cudaSuccess)
    {
        // printf("fail%i\n", code);
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        //if (abort) exit(code);
    }

}

__device__ __host__ int cholesky(double* orig, int n, double* aug, int mcol, double* chol, double* cholaug, int ofs, int d)
{
    int i, j, k, l;
    int retval = 1;

    for (i = ofs; i < n + ofs; i++) {
        chol[d * n * n + i * n + i] = orig[d * n * n + i * n + i];
        for (k = ofs; k < i; k++)
            chol[d * n * n + i * n + i] -= chol[d * n * n + k * n + i] * chol[d * n * n + k * n + i];
        if (chol[d * n * n + i * n + i] <= 0) {
            /*fprintf(stderr, "\nERROR: non-positive definite matrix!\n");*/
            //printf("\nproblem from %d %f\n", i, chol[i * n + i]);
            retval = 0;
            return retval;
        }
        chol[d * n * n + i * n + i] = sqrt(chol[d * n * n + i * n + i]);

        /*This portion multiplies the extra matrix by C^-t */
        for (l = ofs; l < mcol + ofs; l++) {
            cholaug[i * n + l] = aug[i * n + l];
            for (k = ofs; k < i; k++) {
                cholaug[i * n + l] -= cholaug[k * n + l] * chol[d * n * n + k * n + i];
            }
            cholaug[i * n + l] /= chol[d * n * n + i * n + i];
        }

        for (j = i + 1; j < n + ofs; j++) {
            chol[d * n * n + i * n + j] = orig[d * n * n + i * n + j];
            for (k = ofs; k < i; k++)
                chol[d * n * n + i * n + j] -= chol[d * n * n + k * n + i] * chol[d * n * n + k * n + j];
            chol[d * n * n + i * n + j] /= chol[d * n * n + i * n + i];
        }
    }

    return retval;
}

__device__ __host__ double* exponential_isotropic_dev(double* covparms, double* locs, int n, int dim) {

    double nugget = covparms[0] * covparms[2];
    double* locs_scaled = (double*)malloc(sizeof(double) * n * dim);
    // create scaled locations
    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < n; i++) {
            locs_scaled[i * dim + j] = locs[i * dim + j] / covparms[1];
        }
    }
    double* covmat = (double*)malloc(sizeof(double) * n * n);
    // calculate covariances
    for (int i1 = 0; i1 < n; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            // calculate distance
            double d = 0.0;
            for (int j = 0; j < dim; j++) {
                d += pow(locs_scaled[i1 * dim + j] - locs_scaled[i2 * dim + j], 2.0);
            }
            d = sqrt(d);

            // calculate covariance            
            covmat[i2 * n + i1] = covparms[0] * exp(-d);
            covmat[i1 * n + i2] = covmat[i2 * n + i1];

        }
    }

    // add nugget
    for (int i1 = 0; i1 < n; i1++) {
        covmat[i1 * n + i1] += nugget;
    }

    return covmat;
}

__device__ __host__ void solveUpperTriangular(double* A, double* b, int m, double* x, int d, int n) {

    for (int i = m - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < m; ++j) {
            sum += A[d * m * m + i * m + j] * x[d * m + j];
        }
        x[d * m + i] = (b[i] - sum) / A[d * m * m + i * m + i];
    }
    //return x;
}

//' Single parralel on cpu without arma
double* vecchia_Linv(
    double* covparms,
    double* locs,
    double* NNarray,
    double* Linv,
    int n, int m, int dim,
    int end_ind, int start_ind = 1) {



    #pragma omp parallel
    {
        //double* l_Linv = (double*)malloc(sizeof(double) * n * m);
        double* l_Linv = (double*)calloc(n * m, sizeof(double));
        #pragma omp for
        for (int i = start_ind - 1; i < end_ind; i++) {

            //Rcpp::checkUserInterrupt();
            int bsize = std::min(i + 1, m);
            // first, fill in ysub, locsub, and X0 in reverse order
            //double* locsub = (double*)malloc(sizeof(double) * bsize * dim);
            double* locsub = (double*)calloc(bsize * dim, sizeof(double));
            for (int j = bsize - 1; j >= 0; j--) {
                for (int k = 0; k < dim; k++) { locsub[(bsize - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k]; }
            }

            // compute covariance matrix and derivatives and take cholesky
            //arma::mat covmat = p_covfun[0](covparms, locsub);
            double* covmat = exponential_isotropic_dev(covparms, locsub, bsize, dim);

            //double* R = (double*)malloc(sizeof(double) * bsize * bsize);
            //double* I = (double*)malloc(sizeof(double) * bsize * bsize);
            double* R = (double*)calloc(bsize * bsize, sizeof(double));
            double* I = (double*)calloc(bsize * bsize, sizeof(double));
            for (int da = 0; da < bsize; da++) {
                I[da + bsize * da] = 1;
            }

            int checkchol = cholesky(covmat, bsize, I, bsize, R, I, 0, 0);
            // assert(checkchol != 0);

            //double* onevec = (double*)malloc(sizeof(double) * bsize);
            double* onevec = (double*)calloc(bsize, sizeof(double));
            onevec[bsize - 1] = 1.0;
            //double* choli2 = (double*)malloc(sizeof(double) * bsize);
            double* choli2 = (double*)calloc(bsize, sizeof(double));
            if (checkchol == 0) {
                choli2 = onevec;
                //Rcout << "failed chol" << endl;
                //Rcout << checkchol << endl;
            }
            else {
                solveUpperTriangular(R, onevec, bsize, choli2, 0, 0);
            }
            /*if (i == 2) {
                for (int j = 0; j < bsize; j++) {
                    printf("%f ", choli2[j]);
                }
            }*/
            for (int j = bsize - 1; j >= 0; j--) {
                l_Linv[i * m + bsize - 1 - j] = choli2[j];
            }
        }
        #pragma omp critical
        for (int t = 0; t < end_ind; t++) {
            for (int r = 0; r < m; r++) {
                Linv[t * m + r] = l_Linv[t * m + r];
            }
        }
    }
    /*NumericMatrix Linv_r = wrap(Linv);*/
    return Linv;
}

__global__ void call_vecchia_Linv_gpu(double* locs, double* NNarray,
     double variance, double range, double nugget, double* covmat, double* locs_scaled, int n, int m, int dim, int start_ind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || i < start_ind) {
        return;
    }

    if (m <= 51) {
        double ls[51 * 2];
        double cov[51 * 51];
        
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < dim; k++) {
                // locs_scaled[i * m * dim + (m - 1 - j) * dim + k] = ( locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] ) / range;
                ls[(m - 1 -j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
            }
        }
        
        for (int i1 = 0; i1 < m; i1++) {
            for (int i2 = 0; i2 <= i1; i2++) {
                // calculate distance
                //double d = hypot(locs_scaled[i * m * dim + i1 * dim ]- locs_scaled[i * m * dim + i2 * dim], 
                //locs_scaled[i * m * dim + i1 * dim + 1] - locs_scaled[i * m * dim + i2 * dim + 1]);
                
                //double d = hypot(ls[i1 * dim] - ls[i2 * dim],
                    //ls[i1 * dim + 1] - ls[i2 * dim + 1]);
                
                double d = 0;
                double d2 = 0;
                for (int j = 0; j < dim; j++) {
                    d2 = ls[i1 * dim + j] - ls[i2 * dim + j];
                    d += d2 * d2;
                }
                d = sqrt(d);

                // calculate covariance
                if (i1 == i2) {
                    // covmat[i * m * m + i2 * m + i1] = variance * (expf(-d) + nugget);
                    cov[i2 * m + i1] = variance * (exp(-d) + nugget);
                }
                else {
                    // covmat[i * m * m + i2 * m + i1] = variance * expf(-d);
                    cov[i2 * m + i1] = variance * exp(-d);
                    cov[i1 * m + i2] = cov[i2 * m + i1];
                    // covmat[i * m * m + i1 * m + i2] = covmat[i * m * m + i2 * m + i1];
                }
            }
        }
        
        int k, q, j;
        double temp, diff;
        for (q = 0; q < m; q++) {
            diff = 0;
            for (k = 0; k < m; k++) {
                if (k < q) {
                    temp = cov[k * m + q];
                    // temp = covmat[i * m * m + k * m + q];
                    diff -= temp * temp;
                }
                else if (k == q) {
                    cov[q * m + q] = sqrt(cov[q * m + q] + diff);
                    //covmat[i * m * m + q * m + q] = sqrt(covmat[i * m * m + q * m + q] + diff);;
                    diff = 0;
                }
                else {
                    diff = 0;
                    for (int p = 0; p < q; p++) {
                        //diff += covmat[i * m * m + p * m + q] * covmat[i * m * m + p * m + k];
                        diff += cov[p * m + q] * cov[p * m + k];
                    }
                    //covmat[i * m * m + q * m + k] = (covmat[i * m * m + q * m + k] - diff ) / covmat[i * m * m + q * m + q];
                    cov[q * m + k] = (cov[q * m + k] - diff) / cov[q * m + q];
                }
            }

        }
        int b = 1;
        for (int q = m - 1; q >= 0; q--) {
            double sum = 0.0;
            for (int j = q + 1; j < m; j++) {
                //sum += covmat[i * m * m + q * m + j] * NNarray[i * m + m - 1 - j];
                sum += cov[q * m + j] * NNarray[i * m + m - 1 - j];
            }
            //NNarray[i * m + m - 1 - q] = (b - sum) / covmat[i * m * m + q * m + q];
            NNarray[i * m + m - 1 - q] = (b - sum) / cov[q * m + q];
            b = 0;
        }
    } else if (m <= 51) {
        double ls[51 * 2];
        double cov[51 * 51];
        
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < dim; k++) {
                // locs_scaled[i * m * dim + (m - 1 - j) * dim + k] = ( locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] ) / range;
                ls[(m - 1 -j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
            }
        }
        
        for (int i1 = 0; i1 < m; i1++) {
            for (int i2 = 0; i2 <= i1; i2++) {
                // calculate distance
                //double d = hypot(locs_scaled[i * m * dim + i1 * dim ]- locs_scaled[i * m * dim + i2 * dim], 
                //locs_scaled[i * m * dim + i1 * dim + 1] - locs_scaled[i * m * dim + i2 * dim + 1]);
                
                //double d = hypot(ls[i1 * dim] - ls[i2 * dim],
                    //ls[i1 * dim + 1] - ls[i2 * dim + 1]);
                
                double d = 0;
                double d2 = 0;
                for (int j = 0; j < dim; j++) {
                    d2 = ls[i1 * dim + j] - ls[i2 * dim + j];
                    d += d2 * d2;
                }
                d = sqrt(d);

                // calculate covariance
                if (i1 == i2) {
                    // covmat[i * m * m + i2 * m + i1] = variance * (expf(-d) + nugget);
                    cov[i2 * m + i1] = variance * (exp(-d) + nugget);
                }
                else {
                    // covmat[i * m * m + i2 * m + i1] = variance * expf(-d);
                    cov[i2 * m + i1] = variance * exp(-d);
                    cov[i1 * m + i2] = cov[i2 * m + i1];
                    // covmat[i * m * m + i1 * m + i2] = covmat[i * m * m + i2 * m + i1];
                }
            }
        }
        
        int k, q, j;
        double temp, diff;
        for (q = 0; q < m; q++) {
            diff = 0;
            for (k = 0; k < m; k++) {
                if (k < q) {
                    temp = cov[k * m + q];
                    // temp = covmat[i * m * m + k * m + q];
                    diff -= temp * temp;
                }
                else if (k == q) {
                    cov[q * m + q] = sqrt(cov[q * m + q] + diff);
                    //covmat[i * m * m + q * m + q] = sqrt(covmat[i * m * m + q * m + q] + diff);;
                    diff = 0;
                }
                else {
                    diff = 0;
                    for (int p = 0; p < q; p++) {
                        //diff += covmat[i * m * m + p * m + q] * covmat[i * m * m + p * m + k];
                        diff += cov[p * m + q] * cov[p * m + k];
                    }
                    //covmat[i * m * m + q * m + k] = (covmat[i * m * m + q * m + k] - diff ) / covmat[i * m * m + q * m + q];
                    cov[q * m + k] = (cov[q * m + k] - diff) / cov[q * m + q];
                }
            }

        }
        int b = 1;
        for (int q = m - 1; q >= 0; q--) {
            double sum = 0.0;
            for (int j = q + 1; j < m; j++) {
                //sum += covmat[i * m * m + q * m + j] * NNarray[i * m + m - 1 - j];
                sum += cov[q * m + j] * NNarray[i * m + m - 1 - j];
            }
            //NNarray[i * m + m - 1 - q] = (b - sum) / covmat[i * m * m + q * m + q];
            NNarray[i * m + m - 1 - q] = (b - sum) / cov[q * m + q];
            b = 0;
        }

    }
    

}

// Calculates the sparse inverse Cholesky matrix by calling a kernel function
extern "C"
double* vecchia_Linv_gpu_outer(
    double* covparms,
    double* locs,
    double* NNarray,
    int n,
    int m,
    int dim) {

    int start_ind = 1;

    double* Linv = (double*)calloc(m * m, sizeof(double));
    vecchia_Linv(covparms, locs, NNarray, Linv, n, m, dim, m);
    // Allocate device memory
    double * d_locs, * d_covmat ,*d_locs_scaled;
    double *d_NNarray;
    //auto begin = std::chrono::steady_clock::now();
    gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
    gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    /*gpuErrchk(cudaMalloc((void**)&d_covmat, sizeof(double) * n * m * m));
    gpuErrchk(cudaMalloc((void**)&d_locs_scaled, sizeof(double) * n * m * dim))*/;

    gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));

    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "Memcpy Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[mus]" << std::endl;


    int grid_size = 16;
    int block_size = ((n + grid_size) / grid_size);
    call_vecchia_Linv_gpu <<<block_size, grid_size >>> (d_locs, d_NNarray, covparms[0], covparms[1], covparms[2], d_covmat, d_locs_scaled, n, m, dim, m);
    //call_vecchia_Linv_gpu_double << <NUM_BLOCKS, dim3(m,m) >> > (d_locs, d_NNarray, covparms[0], covparms[1], covparms[2], d_covmat, d_locs_scaled, n, m, dim, m);
    //call_vecchia_Linv_gpu_double_com << <NUM_BLOCKS, dim3(m,m) >> > (d_locs, d_NNarray, covparms[0], covparms[1], covparms[2], d_covmat, d_locs_scaled, n, m, dim, m);
    //call_vecchia_Linv_gpu_double_conj << <NUM_BLOCKS, dim3(m,m) >> > (d_locs, d_NNarray, covparms[0], covparms[1], covparms[2], d_covmat, d_locs_scaled, n, m, dim, m);
    //call_vecchia_Linv_gpu_single_precision << <block_size, grid_size >> > (d_locs, d_NNarray, covparms[0], covparms[1], covparms[2], d_covmat, d_locs_scaled, n, m, dim, m);

    cudaDeviceSynchronize();


    //begin = std::chrono::steady_clock::now();
    gpuErrchk(cudaMemcpy(NNarray, d_NNarray, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
    //end = std::chrono::steady_clock::now();
    //std::cout << "Memcpy final answer Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[mus]" << std::endl;

    for (int r = 0; r < m; r++) {
        for (int s = 0; s < m; s++) {
            NNarray[r * m + s] = Linv[r * m + s];
        }
    }

    return NNarray;


}

// Substitutes locations for NNarray indicies
// returns n array contianing pointers to m * dim array
__global__ void substitute_batched_kernel(double* NNarray, double* locs, double* sub_locs, double* covparms, int n, int m, int dim) {
    //int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = threadIdx.y;
	
	if (i > n || j > m || k > dim) {
        return;
    }

    sub_locs[i * m * dim + (m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / covparms[1];


}

// Calculates covariances
// returns n * m * m array
__global__ void covariance_batched_kernel(double* sub_locs, double* cov, double* covparms, int n, int m, int dim) {
    int i = blockIdx.x;
    int i1 = threadIdx.x;
    int i2 = threadIdx.y;
    if (i < m || i1 > m || i2 > m) {
        return;
    }
    if (i1 == i2) {
        cov[i * m * m + i1 * m + i2] = covparms[0] * (1 + covparms[2]);
        return;
    }
    double d = 0;
    double temp;
    for (int k = 0; k < dim; k++) {
        temp = sub_locs[i * m * dim + i1 * dim + k] - sub_locs[i * m * dim + i2 * dim + k];
        d += temp * temp;
    }
    d = sqrt(d);
    
    cov[i * m * m + i1 * m + i2] = exp(-d) * covparms[0];
    cov[i * m * m + i2 * m + i1] = cov[i * m * m + i1 * m + i2];
}

__global__ void cholesky_and_solve_batched_kernel(double* NNarray, double* covmat, double* covparms, int n, int m, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m || i > n) {
        return;
    }
    int k, q, j;
    double temp, diff;
    for (q = 0; q < m; q++) {
        diff = 0;
        for (k = 0; k < m; k++) {
            if (k < q) {
                temp = covmat[i * m * m + k * m + q];
                diff -= temp * temp;
            }
            else if (k == q) {
                covmat[i * m * m + q * m + q] = sqrt(covmat[i * m * m + q * m + q] + diff);
                diff = 0;
            }
            else {
                diff = 0;
                for (int p = 0; p < q; p++) {
                    diff += covmat[i * m * m + p * m + q] * covmat[i * m * m + p * m + k];
                }
                covmat[i * m * m + q * m + k] = (covmat[i * m * m + q * m + k] - diff ) / covmat[i * m * m + q * m + q];
            }
        }

    }

    int b = 1;
    for (int q = m - 1; q >= 0; q--) {
        double sum = 0.0;
        for (int j = q + 1; j < m; j++) {
            sum += covmat[i * m * m + q * m + j] * NNarray[i * m + m - 1 - j];
        }
        NNarray[i * m + m - 1 - q] = (b - sum) / covmat[i * m * m + q * m + q];
        b = 0;
    }
}

extern "C"
double* vecchia_Linv_gpu_batched(
    double* covparms,
    double* locs,
    double* NNarray,
    int n,
    int m,
    int dim) {

    double* Linv = (double*)calloc(m * m, sizeof(double));
    vecchia_Linv(covparms, locs, NNarray, Linv, n, m, dim, m);


    //transfer to gpu
    double* d_covparms, * d_locs, * d_NNarray;
    gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
    gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_covparms, sizeof(double) * 3));


    cudaMemcpy(d_covparms, covparms, sizeof(double*) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_NNarray, NNarray, sizeof(double*) * n * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_locs, locs, sizeof(double*) * n * dim, cudaMemcpyHostToDevice);

    double *d_sublocs, *d_cov;
    gpuErrchk(cudaMalloc((void**)&d_sublocs, sizeof(double) * n * m * dim));
    gpuErrchk(cudaMalloc((void**)&d_cov, sizeof(double) * n * m * m))
    
    
    dim3 threadsPerBlock(m, dim);
    int numBlocks = n; //(n + 32 - 1) / 32;
    substitute_batched_kernel << <numBlocks, threadsPerBlock >> > (d_NNarray, d_locs, d_sublocs, d_covparms, n, m, dim);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock2(m, m);
    int numBlocks2 = n;
    covariance_batched_kernel << <numBlocks2, threadsPerBlock2 >> > (d_sublocs, d_cov, d_covparms, n, m, dim);
    cudaDeviceSynchronize();

    int threadsPerBlock3 = 32;
	int numBlocks3 = (n + 32 - 1) / 32;
    cholesky_and_solve_batched_kernel << <numBlocks3, threadsPerBlock3 >> > (d_NNarray, d_cov, d_covparms, n, m, dim);
    cudaDeviceSynchronize();

    double* vecchia_Linv = (double*)malloc(sizeof(double) * n * m);
    cudaMemcpy(vecchia_Linv, d_NNarray, sizeof(double) * n * m, cudaMemcpyDeviceToHost);
    for (int r = 0; r < m; r++) {
        for (int s = 0; s < m; s++) {
            vecchia_Linv[r * m + s] = Linv[r * m + s];
        }
    }

    return vecchia_Linv;

}


void compute_pieces_cpu(
    double variance, double range, double nugget,
    double* locs,
    double* NNarray,
    double* y,
    double* X,
    double* XSX,
    double* ySX,
    double ySy,
    double logdet,
    double* dXSX,
    double* dySX,
    double* dySy,
    double* dlogdet,
    double* ainfo,
    int profbeta,
    int grad_info,
    int n, int m, int p, int nparms, int dim
) {

    
   	// printf("Inside\n");

//     /*arma::mat l_XSX = arma::mat(p, p, arma::fill::zeros);
//     arma::vec l_ySX = arma::vec(p, arma::fill::zeros);
//     double l_ySy = 0.0;
//     double l_logdet = 0.0;
//     arma::cube l_dXSX = arma::cube(p, p, nparms, arma::fill::zeros);
//     arma::mat l_dySX = arma::mat(p, nparms, arma::fill::zeros);
//     arma::vec l_dySy = arma::vec(nparms, arma::fill::zeros);
//     arma::vec l_dlogdet = arma::vec(nparms, arma::fill::zeros);
//     arma::mat l_ainfo = arma::mat(nparms, nparms, arma::fill::zeros);*/
    // printf("n: %f", variance);
    for (int i = 0; i < n; i++) {
        // printf("%i", i);
        clock_t start = clock();
        int bsize = std::min(i + 1, m);
        // if (bsize == m) {
        //     continue;
        // }
        double* locsub = (double*)malloc(sizeof(double) * bsize * dim);
        double* ysub = (double*)malloc(sizeof(double) * bsize );
        double* X0 = (double*)malloc(sizeof(double) * bsize * p);
        for (int j = bsize - 1; j >= 0; j--) {
            ysub[bsize - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
            for (int k = 0; k < dim; k++) {
                locsub[(bsize - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
            }
            if (profbeta) {
                for (int k = 0; k < p; k++) {
                    X0[(bsize - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
                }
            }
        }
       
        clock_t end = clock();
        // if (i == 10000) {
        //     printf("Substituting NN + scaling: %i\n", end - start);
        // }
        start = clock();
        // compute covariance matrix and derivatives and take cholesky
        // arma::mat covmat = p_covfun[0](covparms, locsub);
        //double* covmat = exponential_isotropic(covparms, locsub, m, dim);
        // Calculate covmatrix
        double* covmat = (double*)malloc(sizeof(double) * bsize * bsize);
        
        double temp;
        for (int i1 = 0; i1 < bsize; i1++) {
            for (int i2 = 0; i2 <= i1; i2++) {
                double d = 0.0;
                for (int j = 0; j < dim; j++) {
                    temp = locsub[i1 * dim + j] - locsub[i2 * dim + j];
                    d += temp * temp;
                }
                d = sqrt(d);
                // calculate covariance
                if (i1 == i2) {
                    covmat[i2 * bsize + i1] = variance * (exp(-d) + nugget);
                }
                else {
                    covmat[i2 * bsize + i1] = variance * exp(-d);
                    covmat[i1 * bsize + i2] = covmat[i2 * bsize + i1];
                }
            }
        }
    
        end = clock();
        /*if (i == 40) {
            printf("GPU covmat\n");
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    printf("%f ", covmat[40 * m * m + i * m + j]);
                }
                printf("\n");
            }
        }*/
        // if (i == 10000) {
        //     printf("Covariance matrix: %i\n", end - start);
        // }
        
        start = clock();
        double* dcovmat = (double*)malloc(sizeof(double) * bsize * bsize * nparms);
    
        if (grad_info) {
            // calculate derivatives
            //arma::cube dcovmat = arma::cube(n, n, covparms.n_elem, fill::zeros);
            //dcovmat = (double*)malloc(sizeof(double) * m * m * nparms);
            for (int i1 = 0; i1 < bsize; i1++) {
                for (int i2 = 0; i2 <= i1; i2++) {
                    double d = 0.0;
                    double a = 0;
                    for (int j = 0; j < dim; j++) {
                        a = locsub[i1 * dim + j] - locsub[i2 * dim + j];
                        d += a * a;
                    }
                    d = sqrt(d);
                    temp = exp(-d);

                    dcovmat[i1 * bsize * nparms + i2 * nparms + 0] += temp;
                    dcovmat[i1 * bsize * nparms + i2 * nparms + 1] += variance * temp * d / range;
                    if (i1 == i2) { // update diagonal entry
                        dcovmat[i1 * bsize * nparms + i2 * nparms + 0] += nugget;
                        dcovmat[i1 * bsize * nparms + i2 * nparms + 2] = variance;
                    }
                    else { // fill in opposite entry
                        for (int j = 0; j < nparms; j++) {
                            dcovmat[i2 * bsize * nparms + i1 * nparms + j] = dcovmat[i1 * bsize * nparms + i2 * nparms + j];
                        }
                    }
                }
            }

        }
    
        // end = clock();
        // if (i == 10000) {
        //     printf("Covariance derivative: %i\n", end - start);
        // }
        // start = clock();
        /*arma::mat cholmat = eye(size(covmat));
        chol(cholmat, covmat, "lower");*/

        // Cholesky decomposition
        double temp2;
        double diff;
        
        
        int r, j, k, l;
        for (r = 0; r < bsize; r++) {
            diff = 0;
            for (k = 0; k < r; k++) {
                temp = covmat[r * bsize + k];
                diff += temp * temp;
            }
            covmat[r * bsize + r] = sqrt(covmat[r * bsize + r] - diff);


            for (j = r + 1; j < bsize; j++) {
                diff = 0;
                for (k = 0; k < r; k++) {
                    diff += covmat[r * bsize + k] * covmat[j * bsize + k];
                }
                covmat[j * bsize + r] = (covmat[j * bsize + r] - diff) / covmat[r * bsize + r];
            }
        }

        // end = clock();
        // if (i == 10000) {
        //     printf("Cholesky decomposition: %i\n", end - start);
        // }


        start = clock();
        // i1 is conditioning set, i2 is response        
        //arma::span i1 = span(0,bsize-2);
        //arma::span i2 = span(bsize - 1, bsize - 1);

        // get last row of cholmat
        /*arma::vec onevec = zeros(bsize);
        onevec(bsize - 1) = 1.0;*/
        double* choli2 = (double*)malloc(sizeof(double) * bsize);

        if (grad_info) {
            //choli2 = backward_solve(cholmat, onevec, m);
            choli2[bsize - 1] = 1 / covmat[(bsize - 1) * bsize + bsize - 1];

            for (int k = bsize - 2; k >= 0; k--) {
                double dd = 0.0;
                for (int j = bsize - 1; j > k; j--) {
                    dd += covmat[j * bsize + k] * choli2[j];
                }
                choli2[k] = (-dd) / covmat[k * bsize + k];
            }
        }
        // end = clock();
        // if (i == 10000) {
        //     printf("solve(cholmat, onevec, m): %i\n", end - start);
        // }
        bool cond = bsize > 1;
        
        // do solves with X and y
        double* LiX0 = (double*)malloc(sizeof(double) * bsize * p);
    

        if (profbeta) {
            start = clock();
            // LiX0 = forward_solve_mat(covmat, X0, m, p);
            for (int k = 0; k < p; k++) {
                LiX0[0 * p + k] = X0[0 * p + k] / covmat[0 * bsize + 0];
            }

            for (int h = 1; h < bsize; h++) {
                for (int k = 0; k < p; k++) {
                    double dd = 0.0;
                    for (int j = 0; j < h; j++) {
                        dd += covmat[h * bsize + j] * LiX0[j * p + k];
                    }
                    LiX0[h * p + k] = (X0[h * p + k] - dd) / covmat[h * bsize + h];
                }
            }
            // end = clock();
            // if (i == 10000) {
            //     printf("forward_solve_mat(cholmat, X0, m, p): %i\n", end - start);
            // }

        }
    
        for (int j = 0; j < bsize; j++) {
            for (int k = j + 1; k < bsize; k++) {
                covmat[j * bsize + k] = 0.0;
            }
        }
        
        
        //arma::vec Liy0 = solve( trimatl(cholmat), ysub );
        //double* Liy0 = forward_solve(cholmat, ysub, m);
        //double* Liy0 = (double*)malloc(sizeof(double) * m);
        /*for (int j = 0; j < m; j++) {
            Liy0[i * m + j] = 0.0f;
        }*/
        start = clock();
        double* Liy0 = (double*)malloc(sizeof(double) * bsize);

        Liy0[0] = ysub[0] / covmat[0 * bsize + 0];

        for (int k = 1; k < bsize; k++) {
            double dd = 0.0;
            for (int j = 0; j < k; j++) {
                dd += covmat[k * bsize + j] * Liy0[j];
            }
            Liy0[k] = (ysub[k] - dd) / covmat[k * bsize + k];
        }
        // end = clock();
        // if (i == 10000) {
        //     printf("forward_solve(cholmat, ysub, m): %i\n", end - start);
        //     start = clock();
        // }

        // loglik objects
        logdet += 2.0 * log(covmat[(bsize - 1) * bsize + bsize - 1]);

        temp = Liy0[bsize - 1];
        ySy += temp * temp;
    

        
        if (profbeta) {
            start = clock();
            /*l_XSX += LiX0.rows(i2).t() * LiX0.rows(i2);
            l_ySX += (Liy0(i2) * LiX0.rows(i2)).t();*/
            temp2 = Liy0[bsize - 1];
            for (int i1 = 0; i1 < p; i1++) {
                temp = LiX0[(bsize - 1) * p + i1];
                for (int i2 = 0; i2 <= i1; i2++) {
                    XSX[i1 * p + i2] = temp * LiX0[(bsize - 1) * p + i2];
                    XSX[i2 * p + i1] = XSX[i1 * p + i2];
                }
                ySX[i1] = temp2 * LiX0[(bsize - 1) * p + i1];
            }
            // end = clock();
            // if (i == 10000) {
            //     printf("Liy0(i2) * LiX0.rows(i2): %i\n", end - start);
            //     start = clock();
            // }
        }
        
        if (grad_info) {
            // gradient objects
            // LidSLi3 is last column of Li * (dS_j) * Lit for 1 parameter i
            // LidSLi2 stores these columns in a matrix for all parameters
            // arma::mat LidSLi2(bsize, nparms);

            double* LidSLi2 = (double*)malloc(sizeof(double) * bsize * nparms);
            
            if (cond) {
                start = clock();
                for (int j = 0; j < nparms; j++) {
                    double* LidSLi3 = (double*)malloc(sizeof(double) * bsize);
                    double* c = (double*)malloc(sizeof(double) * bsize);

                    // compute last column of Li * (dS_j) * Lit
                    //arma::vec LidSLi3 = forward_solve(cholmat, dcovmat.slice(j) * choli2);
                    // c = dcovmat.slice(j) * choli2
                    for (int h = 0; h < bsize; h++) {
                        c[h] = 0;
                        temp = 0;
                        for (int k = 0; k < bsize; k++) {
                            temp += dcovmat[h * bsize * nparms + k * nparms + j] * choli2[k];
                        }
                        c[h] = temp;
                    }

                    //LidSLi3 = forward_solve(cholmat, c);      
                    LidSLi3[0] = c[0] / covmat[0 * bsize + 0];

                    for (int k = 1; k < bsize; k++) {
                        double dd = 0.0;
                        for (int l = 0; l < k; l++) {
                            dd += covmat[k * bsize + l] * LidSLi3[l];
                        }
                        LidSLi3[k] = (c[k] - dd) / covmat[k * bsize + k];
                    }

                    ////////////////
                    //arma::vec v1 = LiX0.t() * LidSLi3;
                    double* v1 = (double*)malloc(sizeof(double) * p);

                    for (int h = 0; h < p; h++) {
                        v1[h] = 0;
                        temp = 0;
                        for (int k = 0; k < bsize; k++) {
                            temp += LiX0[k * p + h] * LidSLi3[k];
                        }
                        v1[h] = temp;
                    }
                   
                    ////////////////

                    //double s1 = as_scalar(Liy0.t() * LidSLi3);
                    double s1 = 0;
                    for (int h = 0; h < bsize; h++) {
                        s1 += Liy0[h] * LidSLi3[h];
                    }

                    ////////////////

                    /*(l_dXSX).slice(j) += v1 * LiX0.rows(i2) + (v1 * LiX0.rows(i2)).t() -
                        as_scalar(LidSLi3(i2)) * (LiX0.rows(i2).t() * LiX0.rows(i2));*/

                        //double* v1LiX0 = (double*)malloc(sizeof(double) * m * m);
                    double temp3;
                    double temp4 = LidSLi3[bsize - 1];
                    for (int h = 0; h < p; h++) {
                        temp = v1[h];
                        temp2 = LiX0[(bsize - 1) * p + h];

                        for (int k = 0; k < p; k++) {
                            temp3 = LiX0[(m - 1) * p + k];
                            dXSX[h * p * nparms + k * nparms + j] += temp * temp3 +
                                (v1[k] - temp4 * temp3) * temp2;
                        }
                    }
                    temp = Liy0[bsize - 1];
                    ///////////////
                    /*(l_dySy)(j) += as_scalar(2.0 * s1 * Liy0(i2) -
                        LidSLi3(i2) * Liy0(i2) * Liy0(i2));*/
                    dySy[j] += (2 * s1 - temp4 * temp) * temp;

                    /*(l_dySX).col(j) += (s1 * LiX0.rows(i2) + (v1 * Liy0(i2)).t() -
                        as_scalar(LidSLi3(i2)) * LiX0.rows(i2) * as_scalar(Liy0(i2))).t();*/
                    temp3 = LidSLi3[bsize - 1];
                    for (int h = 0; h < p; h++) {
                        temp2 = LiX0[(bsize - 1) * p + h];
                        dySX[h * nparms + j] += s1 * temp2 +
                            v1[h] * temp - temp3 * temp2 * temp;
                    }

                    //(l_dlogdet)(j) += as_scalar(LidSLi3(i2));
                    dlogdet[j] += temp3;

                    //LidSLi2.col(j) = LidSLi3;
                    for (int h = 0; h < bsize; h++) {
                        LidSLi2[h * nparms + j] = LidSLi3[h];
                    }
                    /*if (i == 40 && j == 2) {
                        printf("CPU s1\n");
                        printf("%f", s1);
                    }*/

                }
                // end = clock();
                // if (i == 10000) {
                //     printf("gradient objects: %i\n", end - start);
                //     start = clock();
                // }
                // start = clock();
                // fisher information object
                // bottom right corner gets double counted, so subtract it off
                for (int h = 0; h < nparms; h++) {
                    temp2 = LidSLi2[(bsize - 1) * nparms + h];
                    for (int j = 0; j < h + 1; j++) {
                        /*(l_ainfo)(h, j) +=
                            1.0 * accu(LidSLi2.col(h) % LidSLi2.col(j)) -
                            0.5 * accu(LidSLi2.rows(i2).col(j) %
                                LidSLi2.rows(i2).col(h));*/
                        double s = 0;
                        for (int l = 0; l < bsize; l++) {
                            s += LidSLi2[l * bsize + h] * LidSLi2[l * bsize + j];
                        }
                        ainfo[h * nparms + j] = s - 0.5 * LidSLi2[(bsize - 1) * nparms + j] * temp2;
                    }
                }
                // end = clock();
                // if (i == 10000) {
                //     printf("fisher information object: %i\n", end - start);
                //     start = clock();
                // }
                

                
            }
            else {
                // for(int j=0; j<nparms; j++){
                //     arma::mat LidSLi = forward_solve_mat( cholmat, dcovmat.slice(j) );
                //     LidSLi = forward_solve_mat( cholmat, LidSLi.t() );
                //     (l_dXSX).slice(j) += LiX0.t() *  LidSLi * LiX0; 
                //     (l_dySy)(j) += as_scalar( Liy0.t() * LidSLi * Liy0 );
                //     (l_dySX).col(j) += ( ( Liy0.t() * LidSLi ) * LiX0 ).t();
                //     (l_dlogdet)(j) += trace( LidSLi );
                //     LidSLi2.col(j) = LidSLi;
                // }
            
                // // fisher information object
                // for(int i=0; i<nparms; i++){ for(int j=0; j<i+1; j++){
                //     (l_ainfo)(i,j) += 0.5*accu( LidSLi2.col(i) % LidSLi2.col(j) ); 
                // }}
                /////////////////////

                // for(int j=0; j<nparms; j++){
                //     //arma::mat LidSLi = forward_solve_mat( cholmat, dcovmat.slice(j) );
                //     double* LidSLi = (double*) malloc(bsize * bsize * sizeof(double));
                //     for (int k = 0; k < bsize; k++) {
                //         LidSLi[0 * bsize + k] = dcovmat[0 * bsize * bsize + k * bsize + j] / covmat[0 * bsize + 0];
                //     }
                //     for (int h = 1; h < bsize; h++) {
                //         for (int k = 0; k < bsize; k++) {
                //             double dd = 0.0;
                //             for (int j = 0; j < h; j++) {
                //                 dd += covmat[h * bsize + j] * LidSLi[j * bsize + k];
                //             }
                //             LidSLi[h * p + k] = (dcovmat[h * bsize * bsize + k * bsize + j] - dd) / covmat[h * bsize + h];
                //         }
                //     }
                //     //LidSLi = forward_solve_mat( cholmat, LidSLi.t() );
                //     double* LidSLi2 = (double*) malloc(bsize * bsize * sizeof(double));
                //     for (int k = 0; k < bsize; k++) {
                //         LidSLi2[0 * bsize + k] = LidSLi[k * bsize + 0] / covmat[0 * bsize + 0];
                //     }
                //     for (int h = 1; h < bsize; h++) {
                //         for (int k = 0; k < bsize; k++) {
                //             double dd = 0.0;
                //             for (int j = 0; j < h; j++) {
                //                 dd += covmat[h * bsize + j] * LidSLi2[j * bsize + k];
                //             }
                //             LidSLi2[h * p + k] = (LidSLi[k * bsize + h] - dd) / covmat[h * bsize + h];
                //         }
                //     }
                //     //(l_dXSX).slice(j) += LiX0.t() *  LidSLi * LiX0; 
                //     for (r = 0; r < p; r++){
                //         for (t = 0; t < p; t++){
                //             dXSX[]
                //         }
                //     }

                // }


            }
        }


    }
    

}

__global__ void compute_pieces(double* y, double* X, double* NNarray, double* locs, /*double* locsub,*/
    /*double* covmat,*/ double* logdet, double* ySy, double* XSX, double* ySX,
    double* dXSX, double* dySX, double* dySy, double* dlogdet, double* ainfo,
    double variance, double range, double nugget,
    int n, int p, int m, int dim, int nparms,
    bool profbeta, bool grad_info//,
    /*double* dcovmat,
    double* ysub, double* X0, double* Liy0, double* LiX0, double* choli2, double* onevec,
    double* LidSLi2, double* c, double* v1, double* LidSLi3*/) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int bsize = std::min(i + 1, m);
    if (i < m || i >= n) {
        return;
    }
    if (m <= 31) {
        double ysub[31];
        double locsub[31 * 2];
        double X0[31 * 10];
        for (int j = m - 1; j >= 0; j--) {
            // ysub[i * m + m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
            ysub[m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
            for (int k = 0; k < dim; k++) {
                // locsub[i * m * dim + (m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
                locsub[(m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
            }
            if (profbeta) {
                for (int k = 0; k < p; k++) {
                    // X0[i * m * p + (m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
                    X0[(m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
                }
            }
        }
    
        double covmat[31 * 31];
       
        double temp;
        for (int i1 = 0; i1 < m; i1++) {
            for (int i2 = 0; i2 <= i1; i2++) {
                double d = 0.0;
                for (int j = 0; j < dim; j++) {
                    // temp = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                    temp = locsub[i1 * dim + j] - locsub[i2 * dim + j];
                    d += temp * temp;
                }
                d = sqrt(d);
                // calculate covariance
                if (i1 == i2) {
                    // covmat[i * m * m + i2 * m + i1] = variance * (exp(-d) + nugget);
                    covmat[i2 * m + i1] = variance * (exp(-d) + nugget);
                }
                else {
                    // covmat[i * m * m + i2 * m + i1] = variance * exp(-d);
                    // covmat[i * m * m + i1 * m + i2] = covmat[i * m * m + i2 * m + i1];

                    covmat[i2 * m + i1] = variance * exp(-d);
                    covmat[i1 * m + i2] = covmat[i2 * m + i1];
                }
            }
        }
        
        double dcovmat[31 * 31 * 3];
        if (grad_info) {
            // calculate derivatives
            //arma::cube dcovmat = arma::cube(n, n, covparms.n_elem, fill::zeros);
            //dcovmat = (double*)malloc(sizeof(double) * m * m * nparms);
            for (int i1 = 0; i1 < m; i1++) {
                for (int i2 = 0; i2 <= i1; i2++) {
                    dcovmat[i1 * m * nparms + i2 * nparms + 0] = 0;
                    dcovmat[i1 * m * nparms + i2 * nparms + 1] = 0;
                    double d = 0.0;
                    double a = 0;
                    for (int j = 0; j < dim; j++) {
                        // a = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                        a = locsub[i1 * dim + j] - locsub[i2 * dim + j];
                        d += a * a;
                    }
                    d = sqrt(d);
                    temp = exp(-d);

                    // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += temp;
                    // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 1] += variance * temp * d / range;

                    dcovmat[i1 * m * nparms + i2 * nparms + 0] += temp;
                    dcovmat[i1 * m * nparms + i2 * nparms + 1] += variance * temp * d / range;
                    if (i1 == i2) { // update diagonal entry
                        // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += nugget;
                        // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 2] = variance;

                        dcovmat[i1 * m * nparms + i2 * nparms + 0] += nugget;
                        dcovmat[i1 * m * nparms + i2 * nparms + 2] = variance;
                    }
                    else { // fill in opposite entry
                        for (int j = 0; j < nparms; j++) {
                            // dcovmat[i * m * m * nparms + i2 * m * nparms + i1 * nparms + j] = dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + j];
                            dcovmat[i2 * m * nparms + i1 * nparms + j] = dcovmat[i1 * m * nparms + i2 * nparms + j];
                        }
                    }
                }
            }
            
        }
       
        /*arma::mat cholmat = eye(size(covmat));
        chol(cholmat, covmat, "lower");*/

        // Cholesky decomposition
        //int k, q, j;
        double temp2;
        double diff;
    
        int r, j, k, l;
        int retval = 1;
        for (r = 0; r < m + 0; r++) {
            diff = 0;
            for (k = 0; k < r; k++) {
                // temp = covmat[i * m * m + r * m + k];
                temp = covmat[r * m + k];
                diff += temp * temp;
            }
            // covmat[i * m * m + r * m + r] = sqrt(covmat[i * m * m + r * m + r] - diff);
            covmat[r * m + r] = sqrt(covmat[r * m + r] - diff);


            for (j = r + 1; j < m + 0; j++) {
                diff = 0;
                for (k = 0; k < r; k++) {
                    // diff += covmat[i * m * m + r * m + k] * covmat[i * m * m + j * m + k];
                    diff += covmat[r * m + k] * covmat[j * m + k];
                }
                // covmat[i * m * m + j * m + r] = (covmat[i * m * m + j * m + r] - diff) / covmat[i * m * m + r * m + r];
                covmat[j * m + r] = (covmat[j * m + r] - diff) / covmat[r * m + r];
            }
        }

        
        // i1 is conditioning set, i2 is response        
        //arma::span i1 = span(0,bsize-2);
        //arma::span i2 = span(bsize - 1, bsize - 1);

        // get last row of cholmat
        /*arma::vec onevec = zeros(bsize);
        onevec(bsize - 1) = 1.0;*/
        double choli2[31];
        if (grad_info) {
            //choli2 = backward_solve(cholmat, onevec, m);
            // choli2[i * m + m - 1] = 1 / covmat[i * m * m + (m - 1) * m + m - 1];
            choli2[m - 1] = 1 / covmat[(m - 1) * m + m - 1];

            for (int k = m - 2; k >= 0; k--) {
                double dd = 0.0;
                for (int j = m - 1; j > k; j--) {
                    // dd += covmat[i * m * m + j * m + k] * choli2[i * m + j];
                    dd += covmat[j * m + k] * choli2[j];
                }
                // choli2[i * m + k] = (-dd) / covmat[i * m * m + k * m + k];
                choli2[k] = (-dd) / covmat[k * m + k];
            }
        }
        
        //bool cond = bsize > 1;
        double LiX0[31 * 10];
        // do solves with X and y
        if (profbeta) {
            // LiX0 = forward_solve_mat(cholmat, X0, m, p);
            for (int k = 0; k < p; k++) {
                // LiX0[i * m * p + 0 * p + k] = X0[i * m * p + 0 * p + k] / covmat[i * m * m + 0 * m + 0];
                LiX0[0 * p + k] = X0[0 * p + k] / covmat[0 * m + 0];

            }

            for (int h = 1; h < m; h++) {
                for (int k = 0; k < p; k++) {
                    double dd = 0.0;
                    for (int j = 0; j < h; j++) {
                        // dd += covmat[i * m * m + h * m + j] * LiX0[i * m * p + j * p + k];
                        dd += covmat[h * m + j] * LiX0[j * p + k];
                    }
                    // LiX0[i * m * p + h * p + k] = (X0[i * m * p + h * p + k] - dd) / covmat[i * m * m + h * m + h];
                    LiX0[h * p + k] = (X0[h * p + k] - dd) / covmat[h * m + h];
                }
            }
            

        }
        for (int j = 0; j < m; j++) {
            for (int k = j + 1; k < m; k++) {
                // covmat[i * m * m + j * m + k] = 0.0;
                covmat[j * m + k] = 0.0;
            }
        }

        //arma::vec Liy0 = solve( trimatl(cholmat), ysub );
        //double* Liy0 = forward_solve(cholmat, ysub, m);
        //double* Liy0 = (double*)malloc(sizeof(double) * m);
        /*for (int j = 0; j < m; j++) {
            Liy0[i * m + j] = 0.0f;
        }*/
        double Liy0[31];
        // Liy0[i * m + 0] = ysub[i * m + 0] / covmat[i * m * m + 0 * m + 0];
        Liy0[0] = ysub[0] / covmat[0 * m + 0];

        for (int k = 1; k < m; k++) {
            double dd = 0.0;
            for (int j = 0; j < k; j++) {
                // dd += covmat[i * m * m + k * m + j] * Liy0[i * m + j];
                dd += covmat[k * m + j] * Liy0[j];
            }
            // Liy0[i * m + k] = (ysub[i * m + k] - dd) / covmat[i * m * m + k * m + k];
            Liy0[k] = (ysub[k] - dd) / covmat[k * m + k];

        }
       

        // loglik objects
        // logdet[i] = 2.0 * log(covmat[i * m * m + (m - 1) * m + m - 1]);
        logdet[i] = 2.0 * log(covmat[(m - 1) * m + m - 1]);
        
        // temp = Liy0[i * m + m - 1];
        temp = Liy0[m - 1];
        ySy[i] = temp * temp;

        
        if (profbeta) {
            /*l_XSX += LiX0.rows(i2).t() * LiX0.rows(i2);
            l_ySX += (Liy0(i2) * LiX0.rows(i2)).t();*/
            // temp2 = Liy0[i * m + m - 1];
            temp2 = Liy0[m - 1];
            for (int i1 = 0; i1 < p; i1++) {
                // temp = LiX0[i * m * p + (m - 1) * p + i1];
                temp = LiX0[(m - 1) * p + i1];
                for (int i2 = 0; i2 <= i1; i2++) {
                    // XSX[i * p * p + i1 * p + i2] = temp * LiX0[i * m * p + (m - 1) * p + i2];
                    // XSX[i * p * p + i2 * p + i1] = XSX[i * p * p + i1 * p + i2];

                    XSX[i * p * p + i1 * p + i2] = temp * LiX0[(m - 1) * p + i2];
                    XSX[i * p * p + i2 * p + i1] = XSX[i * p * p + i1 * p + i2];
                }
                // ySX[i * p + i1] = temp2 * LiX0[i * m * p + (m - 1) * p + i1];
                ySX[i * p + i1] = temp2 * LiX0[(m - 1) * p + i1];
            }
            
        }
        double LidSLi3[31];
        double c[31];
		double LidSLi2[31 * 3];
        double v1[10];
        if (grad_info) {
            // gradient objects
            // LidSLi3 is last column of Li * (dS_j) * Lit for 1 parameter i
            // LidSLi2 stores these columns in a matrix for all parameters
            // arma::mat LidSLi2(bsize, nparms);



            for (int j = 0; j < nparms; j++) {
                // compute last column of Li * (dS_j) * Lit
                //arma::vec LidSLi3 = forward_solve(cholmat, dcovmat.slice(j) * choli2);
                // c = dcovmat.slice(j) * choli2
                for (int h = 0; h < m; h++) {
                    c[h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        // temp += dcovmat[i * m * m * nparms + h * m * nparms + k * nparms + j] * choli2[i * m + k];
                        temp += dcovmat[h * m * nparms + k * nparms + j] * choli2[k];
                    }
                    // c[i * m + h] = temp;
                    c[h] = temp;
                }

                //LidSLi3 = forward_solve(cholmat, c);      
                // LidSLi3[i * m + 0] = c[i * m + 0] / covmat[i * m * m + 0 * m + 0];
                LidSLi3[0] = c[0] / covmat[0 * m + 0];

                for (int k = 1; k < m; k++) {
                    double dd = 0.0;
                    for (int l = 0; l < k; l++) {
                        // dd += covmat[i * m * m + k * m + l] * LidSLi3[i * m + l];
                        dd += covmat[k * m + l] * LidSLi3[l];
                    }
                    // LidSLi3[i * m + k] = (c[i * m + k] - dd) / covmat[i * m * m + k * m + k];
                    LidSLi3[k] = (c[k] - dd) / covmat[k * m + k];
                }

                ////////////////
                //arma::vec v1 = LiX0.t() * LidSLi3;

                for (int h = 0; h < p; h++) {
                    v1[h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        // temp += LiX0[i * m * p + k * p + h] * LidSLi3[i * m + k];
                        temp += LiX0[k * p + h] * LidSLi3[k];
                    }
                    v1[h] = temp;
                }

                ////////////////

                //double s1 = as_scalar(Liy0.t() * LidSLi3);
                double s1 = 0;
                for (int h = 0; h < m; h++) {
                    // s1 += Liy0[i * m + h] * LidSLi3[i * m + h];
                    s1 += Liy0[h] * LidSLi3[h];
                }

                ////////////////

                /*(l_dXSX).slice(j) += v1 * LiX0.rows(i2) + (v1 * LiX0.rows(i2)).t() -
                    as_scalar(LidSLi3(i2)) * (LiX0.rows(i2).t() * LiX0.rows(i2));*/

                    //double* v1LiX0 = (double*)malloc(sizeof(double) * m * m);
                double temp3;
                // double temp4 = LidSLi3[i * m + m - 1];
                double temp4 = LidSLi3[m - 1];
                for (int h = 0; h < p; h++) {
                    // temp = v1[i * p + h];
                    // temp2 = LiX0[i * m * p + (m - 1) * p + h];

                    temp = v1[h];
                    temp2 = LiX0[(m - 1) * p + h];

                    for (int k = 0; k < p; k++) {
                        // temp3 = LiX0[i * m * p + (m - 1) * p + k];
                        // dXSX[i * p * p * nparms + h * p * nparms + k * nparms + j] = temp * temp3 +
                        //     (v1[i * p + k] - temp4 * temp3) * temp2;

                        temp3 = LiX0[(m - 1) * p + k];
                        dXSX[i * p * p * nparms + h * p * nparms + k * nparms + j] = temp * temp3 +
                            (v1[k] - temp4 * temp3) * temp2;
                    }
                }
                // temp = Liy0[i * m + m - 1];
                temp = Liy0[m - 1];
                ///////////////
                /*(l_dySy)(j) += as_scalar(2.0 * s1 * Liy0(i2) -
                    LidSLi3(i2) * Liy0(i2) * Liy0(i2));*/
                dySy[i * nparms + j] = (2.0 * s1 - temp4 * temp) * temp;

                /*(l_dySX).col(j) += (s1 * LiX0.rows(i2) + (v1 * Liy0(i2)).t() -
                    as_scalar(LidSLi3(i2)) * LiX0.rows(i2) * as_scalar(Liy0(i2))).t();*/

                // temp3 = LidSLi3[i * m + m - 1];
                temp3 = LidSLi3[m - 1];
                for (int h = 0; h < p; h++) {
                    // temp2 = LiX0[i * m * p + (m - 1) * p + h];
                    // dySX[i * p * nparms + h * nparms + j] = s1 * temp2 +
                    //     v1[i * p + h] * temp - temp3 * temp2 * temp;

                    temp2 = LiX0[(m - 1) * p + h];
                    dySX[i * p * nparms + h * nparms + j] = s1 * temp2 +
                        v1[h] * temp - temp3 * temp2 * temp;
                }

                //(l_dlogdet)(j) += as_scalar(LidSLi3(i2));
                dlogdet[i * nparms + j] = temp3;

                //LidSLi2.col(j) = LidSLi3;
                for (int h = 0; h < m; h++) {
                    // LidSLi2[i * m * nparms + h * nparms + j] = LidSLi3[i * m + h];
                    LidSLi2[h * nparms + j] = LidSLi3[h];
                }
                /*if (i == 40 && j == 2) {
                    printf("CPU s1\n");
                    printf("%f", s1);
                }*/

            }
            
            // fisher information object
            // bottom right corner gets double counted, so subtract it off
            for (int h = 0; h < nparms; h++) {
                // temp2 = LidSLi2[i * m * nparms + (m - 1) * nparms + h];
                temp2 = LidSLi2[(m - 1) * nparms + h];
                for (int j = 0; j < h + 1; j++) {
                    /*(l_ainfo)(h, j) +=
                        1.0 * accu(LidSLi2.col(h) % LidSLi2.col(j)) -
                        0.5 * accu(LidSLi2.rows(i2).col(j) %
                            LidSLi2.rows(i2).col(h));*/
                    double s = 0;
                    for (int l = 0; l < m; l++) {
                        // s += LidSLi2[i * m * nparms + l * nparms + h] * LidSLi2[i * m * nparms + l * nparms + j];
                        s += LidSLi2[l * nparms + h] * LidSLi2[l * nparms + j];
                    }
                    // ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[i * m * nparms + (m - 1) * nparms + j] * temp2;
                    ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[(m - 1) * nparms + j] * temp2;
                }
            }
           
        }
    } else if (m <= 51) {
        double ysub[51];
        double locsub[51 * 2];
        double X0[51 * 10];
        for (int j = m - 1; j >= 0; j--) {
            // ysub[i * m + m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
            ysub[m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
            for (int k = 0; k < dim; k++) {
                // locsub[i * m * dim + (m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
                locsub[(m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
            }
            if (profbeta) {
                for (int k = 0; k < p; k++) {
                    // X0[i * m * p + (m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
                    X0[(m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
                }
            }
        }
        
        double covmat[51 * 51];
       
        double temp;
        for (int i1 = 0; i1 < m; i1++) {
            for (int i2 = 0; i2 <= i1; i2++) {
                double d = 0.0;
                for (int j = 0; j < dim; j++) {
                    // temp = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                    temp = locsub[i1 * dim + j] - locsub[i2 * dim + j];
                    d += temp * temp;
                }
                d = sqrt(d);
                // calculate covariance
                if (i1 == i2) {
                    // covmat[i * m * m + i2 * m + i1] = variance * (exp(-d) + nugget);
                    covmat[i2 * m + i1] = variance * (exp(-d) + nugget);
                }
                else {
                    // covmat[i * m * m + i2 * m + i1] = variance * exp(-d);
                    // covmat[i * m * m + i1 * m + i2] = covmat[i * m * m + i2 * m + i1];

                    covmat[i2 * m + i1] = variance * exp(-d);
                    covmat[i1 * m + i2] = covmat[i2 * m + i1];
                }
            }
        }
        

        double dcovmat[51 * 51 * 3];
        if (grad_info) {
            // calculate derivatives
            //arma::cube dcovmat = arma::cube(n, n, covparms.n_elem, fill::zeros);
            //dcovmat = (double*)malloc(sizeof(double) * m * m * nparms);
            for (int i1 = 0; i1 < m; i1++) {
                for (int i2 = 0; i2 <= i1; i2++) {
                    double d = 0.0;
                    double a = 0;
                    for (int j = 0; j < dim; j++) {
                        // a = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                        a = locsub[i1 * dim + j] - locsub[i2 * dim + j];
                        d += a * a;
                    }
                    d = sqrt(d);
                    temp = exp(-d);

                    // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += temp;
                    // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 1] += variance * temp * d / range;

                    dcovmat[i1 * m * nparms + i2 * nparms + 0] += temp;
                    dcovmat[i1 * m * nparms + i2 * nparms + 1] += variance * temp * d / range;
                    if (i1 == i2) { // update diagonal entry
                        // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += nugget;
                        // dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 2] = variance;

                        dcovmat[i1 * m * nparms + i2 * nparms + 0] += nugget;
                        dcovmat[i1 * m * nparms + i2 * nparms + 2] = variance;
                    }
                    else { // fill in opposite entry
                        for (int j = 0; j < nparms; j++) {
                            // dcovmat[i * m * m * nparms + i2 * m * nparms + i1 * nparms + j] = dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + j];
                            dcovmat[i2 * m * nparms + i1 * nparms + j] = dcovmat[i1 * m * nparms + i2 * nparms + j];
                        }
                    }
                }
            }

        }
       
        /*arma::mat cholmat = eye(size(covmat));
        chol(cholmat, covmat, "lower");*/

        // Cholesky decomposition
        //int k, q, j;
        double temp2;
        double diff;
    
        int r, j, k, l;
        int retval = 1;
        for (r = 0; r < m + 0; r++) {
            diff = 0;
            for (k = 0; k < r; k++) {
                // temp = covmat[i * m * m + r * m + k];
                temp = covmat[r * m + k];
                diff += temp * temp;
            }
            // covmat[i * m * m + r * m + r] = sqrt(covmat[i * m * m + r * m + r] - diff);
            covmat[r * m + r] = sqrt(covmat[r * m + r] - diff);


            for (j = r + 1; j < m + 0; j++) {
                diff = 0;
                for (k = 0; k < r; k++) {
                    // diff += covmat[i * m * m + r * m + k] * covmat[i * m * m + j * m + k];
                    diff += covmat[r * m + k] * covmat[j * m + k];
                }
                // covmat[i * m * m + j * m + r] = (covmat[i * m * m + j * m + r] - diff) / covmat[i * m * m + r * m + r];
                covmat[j * m + r] = (covmat[j * m + r] - diff) / covmat[r * m + r];
            }
        }
        double choli2[51];
        if (grad_info) {
            //choli2 = backward_solve(cholmat, onevec, m);
            // choli2[i * m + m - 1] = 1 / covmat[i * m * m + (m - 1) * m + m - 1];
            choli2[m - 1] = 1 / covmat[(m - 1) * m + m - 1];

            for (int k = m - 2; k >= 0; k--) {
                double dd = 0.0;
                for (int j = m - 1; j > k; j--) {
                    // dd += covmat[i * m * m + j * m + k] * choli2[i * m + j];
                    dd += covmat[j * m + k] * choli2[j];
                }
                // choli2[i * m + k] = (-dd) / covmat[i * m * m + k * m + k];
                choli2[k] = (-dd) / covmat[k * m + k];
            }
        }
        
        //bool cond = bsize > 1;
        double LiX0[51 * 10];
        // do solves with X and y
        if (profbeta) {
            // LiX0 = forward_solve_mat(cholmat, X0, m, p);
            for (int k = 0; k < p; k++) {
                // LiX0[i * m * p + 0 * p + k] = X0[i * m * p + 0 * p + k] / covmat[i * m * m + 0 * m + 0];
                LiX0[0 * p + k] = X0[0 * p + k] / covmat[0 * m + 0];

            }

            for (int h = 1; h < m; h++) {
                for (int k = 0; k < p; k++) {
                    double dd = 0.0;
                    for (int j = 0; j < h; j++) {
                        // dd += covmat[i * m * m + h * m + j] * LiX0[i * m * p + j * p + k];
                        dd += covmat[h * m + j] * LiX0[j * p + k];
                    }
                    // LiX0[i * m * p + h * p + k] = (X0[i * m * p + h * p + k] - dd) / covmat[i * m * m + h * m + h];
                    LiX0[h * p + k] = (X0[h * p + k] - dd) / covmat[h * m + h];
                }
            }
            

        }
        for (int j = 0; j < m; j++) {
            for (int k = j + 1; k < m; k++) {
                // covmat[i * m * m + j * m + k] = 0.0;
                covmat[j * m + k] = 0.0;
            }
        }
        double Liy0[51];
        // Liy0[i * m + 0] = ysub[i * m + 0] / covmat[i * m * m + 0 * m + 0];
        Liy0[0] = ysub[0] / covmat[0 * m + 0];

        for (int k = 1; k < m; k++) {
            double dd = 0.0;
            for (int j = 0; j < k; j++) {
                // dd += covmat[i * m * m + k * m + j] * Liy0[i * m + j];
                dd += covmat[k * m + j] * Liy0[j];
            }
            // Liy0[i * m + k] = (ysub[i * m + k] - dd) / covmat[i * m * m + k * m + k];
            Liy0[k] = (ysub[k] - dd) / covmat[k * m + k];

        }
       

        // loglik objects
        // logdet[i] = 2.0 * log(covmat[i * m * m + (m - 1) * m + m - 1]);
        logdet[i] = 2.0 * log(covmat[(m - 1) * m + m - 1]);

        // temp = Liy0[i * m + m - 1];
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
        double LidSLi3[51];
		double LidSLi2[51 * 3];
        double c[51];
        double v1[10];
        if (grad_info) {
            for (int j = 0; j < nparms; j++) {
                for (int h = 0; h < m; h++) {
                    c[h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        // temp += dcovmat[i * m * m * nparms + h * m * nparms + k * nparms + j] * choli2[i * m + k];
                        temp += dcovmat[h * m * nparms + k * nparms + j] * choli2[k];
                    }
                    // c[i * m + h] = temp;
                    c[h] = temp;
                }

                //LidSLi3 = forward_solve(cholmat, c);      
                // LidSLi3[i * m + 0] = c[i * m + 0] / covmat[i * m * m + 0 * m + 0];
                LidSLi3[0] = c[0] / covmat[0 * m + 0];

                for (int k = 1; k < m; k++) {
                    double dd = 0.0;
                    for (int l = 0; l < k; l++) {
                        // dd += covmat[i * m * m + k * m + l] * LidSLi3[i * m + l];
                        dd += covmat[k * m + l] * LidSLi3[l];
                    }
                    // LidSLi3[i * m + k] = (c[i * m + k] - dd) / covmat[i * m * m + k * m + k];
                    LidSLi3[k] = (c[k] - dd) / covmat[k * m + k];
                }
                for (int h = 0; h < p; h++) {
                    v1[h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        // temp += LiX0[i * m * p + k * p + h] * LidSLi3[i * m + k];
                        temp += LiX0[k * p + h] * LidSLi3[k];
                    }
                    v1[h] = temp;
                }
                double s1 = 0;
                for (int h = 0; h < m; h++) {
                    // s1 += Liy0[i * m + h] * LidSLi3[i * m + h];
                    s1 += Liy0[h] * LidSLi3[h];
                }
                double temp3;
                // double temp4 = LidSLi3[i * m + m - 1];
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
                // temp = Liy0[i * m + m - 1];
                temp = Liy0[m - 1];

                dySy[i * nparms + j] = (2.0 * s1 - temp4 * temp) * temp;

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
                    // LidSLi2[i * m * nparms + h * nparms + j] = LidSLi3[i * m + h];
                    LidSLi2[h * nparms + j] = LidSLi3[h];
                }
            }

            for (int h = 0; h < nparms; h++) {
                // temp2 = LidSLi2[i * m * nparms + (m - 1) * nparms + h];
                temp2 = LidSLi2[(m - 1) * nparms + h];
                for (int j = 0; j < h + 1; j++) {
                    double s = 0;
                    for (int l = 0; l < m; l++) {
                        // s += LidSLi2[i * m * nparms + l * nparms + h] * LidSLi2[i * m * nparms + l * nparms + j];
                        s += LidSLi2[l * nparms + h] * LidSLi2[l * nparms + j];
                    }
                    // ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[i * m * nparms + (m - 1) * nparms + j] * temp2;
                    ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[(m - 1) * nparms + j] * temp2;
                }
            }
           
        }
    }
   
}

extern "C"
double** load_data(
    const double* locs,
    const double* NNarray,
    const double* y,
    const double* X,
    int n, int m, int dim, int p, int nparms) {
        
        // printf("load_data\n");
        
        double* d_locs;
        double* d_NNarray;
        double* d_y;
        double* d_X;

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

        gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
        gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
        gpuErrchk(cudaMalloc((void**)&d_y, sizeof(double) * n));
        gpuErrchk(cudaMalloc((void**)&d_X, sizeof(double) * n * p));

        gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_X, X, sizeof(double) * n * p, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        
        double** data_stores = new double*[12];
        data_stores[0] = d_locs;
        data_stores[1] = d_NNarray;
        data_stores[2] = d_y;
        data_stores[3] = d_X;

        data_stores[4] = d_ySX;
        data_stores[5] = d_XSX;
        data_stores[6] = d_ySy;
        data_stores[7] = d_logdet;
        data_stores[8] = d_dXSX;
        data_stores[9] = d_dySX;
        data_stores[10] = d_dySy;
        data_stores[11] = d_dlogdet;
        data_stores[12] = d_ainfo;
        // printf("load_data...done\n");
        return data_stores;
    }

extern "C"
void free_data(double** data_stores) {
    // int num_pointers = 13;  
    // for (int i = 0; i < num_pointers; i++) {
    //     gpuErrchk(cudaFree(data_stores[i]));
    // }
    cudaDeviceReset();
    
}

extern "C"
void call_compute_pieces_gpu(
    const double* covparms,
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
    const int profbeta,
    const int grad_info,
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

    gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
    gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_y, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_X, sizeof(double) * n * p));

    // d_locs = data_stores[0];
    // d_NNarray = data_stores[1];
    // d_y = data_stores[2];
    // d_X = data_stores[3];
    

    // double* d_covmat;
    // double* d_locs_scaled;

    double* d_ySX;
    double* d_XSX;
    double* d_ySy;
    double* d_logdet;
    double* d_dXSX;
    double* d_dySX;
    double* d_dySy;
    double* d_dlogdet;
    double* d_ainfo;

    // double* d_dcovmat;
    // double* d_ysub;
    // double* d_X0;
    // double* d_Liy0;
    // double* d_LiX0;
    // double* d_choli2;
    // double* d_onevec;
    // double* d_LidSLi2;
    // double* d_c;
    // double* d_v1;
    // double* d_LidSLi3;

    /*gpuErrchk(cudaMalloc((void**)&d_covmat, sizeof(double) * n * m * m));
    gpuErrchk(cudaMalloc((void**)&d_locs_scaled, sizeof(double) * n * m * dim));*/
    
    // printf("n=%i m=%i p=%i dim=%i nparms=%i", n, m, p, dim, nparms);
 
	gpuErrchk(cudaMalloc((void**)&d_ySX, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_XSX, sizeof(double) * n * p * p));
    gpuErrchk(cudaMalloc((void**)&d_ySy, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_logdet, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_dXSX, sizeof(double) * n * p * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySX, sizeof(double) * n * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySy, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dlogdet, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_ainfo, sizeof(double) * n * nparms * nparms));

    /*gpuErrchk(cudaMalloc((void**)&d_dcovmat, sizeof(double) * n * m * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_ysub, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_X0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_Liy0, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_LiX0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_choli2, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_onevec, sizeof(double) * m));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi2, sizeof(double) * n * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_c, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_v1, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi3, sizeof(double) * n * m));*/

    gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_X, X, sizeof(double) * n * p, cudaMemcpyHostToDevice));

    int grid_size = 64;
    int block_size = ((n + grid_size) / grid_size);

    compute_pieces <<<block_size, grid_size>>> (d_y, d_X, d_NNarray, d_locs,/* d_locs_scaled,*/
        /*d_covmat,*/ d_logdet, d_ySy, d_XSX, d_ySX,
        d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
        covparms[0], covparms[1], covparms[2],
        n, p, m, dim, nparms,
        profbeta, grad_info//,
        /*d_dcovmat,
        d_ysub, d_X0, d_Liy0, d_LiX0, d_choli2, d_onevec,
        d_LidSLi2, d_c, d_v1, d_LidSLi3*/);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
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
    for (int i = 0; i < n; i++) {
        
        ySy[0] += l_ySy[i];
        logdet[0] += l_logdet[i];
        for (int j = 0; j < p; j++) {
            ySX[j] += l_ySX[i * p + j];
            for (int k = 0; k < p; k++) {
                XSX[j * p + k] += l_XSX[i * p * p + j * p + k];
                for (int l = 0; l < nparms; l++) {
                    dXSX[j * p * nparms + k * nparms + l] += l_dXSX[i * p * p * nparms + j * p * nparms + k * nparms + l];
                    //dXSX[j * p * nparms + k * nparms + l] += 0;
                }
            }
            for (int k = 0; k < nparms; k++) {
                dySX[j * nparms + k] += l_dySX[i * p * nparms + j * nparms + k];
                //printf("%f ", l_dySX[i * p * nparms + j * nparms + k]);
            }
        }
        for (int j = 0; j < nparms; j++) {
            dySy[j] += l_dySy[i * nparms + j];
            dlogdet[j] += l_dlogdet[i * nparms + j];
            for (int k = 0; k < nparms; k++) {
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

extern "C"
void call_compute_pieces_fisher_gpu(
    const double* covparms,
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
    const int profbeta,
    const int grad_info,
    const int n,
    const int m,
    const int p,
    const int nparms,
    const int dim,
    double** data_store
) {
    
    // printf("call_compute_pieces_fisher_gpu\n");
    double* d_locs;
    double* d_NNarray;
    double* d_y;
    double* d_X;

    // gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
    // gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    // gpuErrchk(cudaMalloc((void**)&d_y, sizeof(double) * n));
    // gpuErrchk(cudaMalloc((void**)&d_X, sizeof(double) * n * p));

    d_locs = data_store[0];
    d_NNarray = data_store[1];
    d_y = data_store[2];
    d_X = data_store[3];
    

    // double* d_covmat;
    // double* d_locs_scaled;

    double* d_ySX = data_store[4];
    double* d_XSX = data_store[5];
    double* d_ySy = data_store[6];
    double* d_logdet = data_store[7];
    double* d_dXSX = data_store[8];
    double* d_dySX = data_store[9];
    double* d_dySy = data_store[10];
    double* d_dlogdet = data_store[11];
    double* d_ainfo = data_store[12];

    // double* d_dcovmat;
    // double* d_ysub;
    // double* d_X0;
    // double* d_Liy0;
    // double* d_LiX0;
    // double* d_choli2;
    // double* d_onevec;
    // double* d_LidSLi2;
    // double* d_c;
    // double* d_v1;
    // double* d_LidSLi3;

    /*gpuErrchk(cudaMalloc((void**)&d_covmat, sizeof(double) * n * m * m));
    gpuErrchk(cudaMalloc((void**)&d_locs_scaled, sizeof(double) * n * m * dim));*/
    
    // printf("n=%i m=%i p=%i dim=%i nparms=%i", n, m, p, dim, nparms);
 
	gpuErrchk(cudaMalloc((void**)&d_ySX, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_XSX, sizeof(double) * n * p * p));
    gpuErrchk(cudaMalloc((void**)&d_ySy, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_logdet, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_dXSX, sizeof(double) * n * p * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySX, sizeof(double) * n * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySy, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dlogdet, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_ainfo, sizeof(double) * n * nparms * nparms));

    /*gpuErrchk(cudaMalloc((void**)&d_dcovmat, sizeof(double) * n * m * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_ysub, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_X0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_Liy0, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_LiX0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_choli2, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_onevec, sizeof(double) * m));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi2, sizeof(double) * n * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_c, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_v1, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi3, sizeof(double) * n * m));*/

    // gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_X, X, sizeof(double) * n * p, cudaMemcpyHostToDevice));

    int grid_size = 64;
    int block_size = ((n + grid_size) / grid_size);

    compute_pieces <<<block_size, grid_size>>> (d_y, d_X, d_NNarray, d_locs,/* d_locs_scaled,*/
        /*d_covmat,*/ d_logdet, d_ySy, d_XSX, d_ySX,
        d_dXSX, d_dySX, d_dySy, d_dlogdet, d_ainfo,
        covparms[0], covparms[1], covparms[2],
        n, p, m, dim, nparms,
        profbeta, grad_info//,
        /*d_dcovmat,
        d_ysub, d_X0, d_Liy0, d_LiX0, d_choli2, d_onevec,
        d_LidSLi2, d_c, d_v1, d_LidSLi3*/);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
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

    // gpuErrchk(cudaFree(d_ySX));
    // gpuErrchk(cudaFree(d_XSX));
    // gpuErrchk(cudaFree(d_ySy));
    // gpuErrchk(cudaFree(d_logdet));
    // gpuErrchk(cudaFree(d_dXSX));
    // gpuErrchk(cudaFree(d_dySX));
    // gpuErrchk(cudaFree(d_dySy));
    // gpuErrchk(cudaFree(d_dlogdet));
    // gpuErrchk(cudaFree(d_ainfo));

    // gpuErrchk(cudaFree(d_locs));
    // gpuErrchk(cudaFree(d_NNarray));
    // gpuErrchk(cudaFree(d_y));
    // gpuErrchk(cudaFree(d_X));

    ySy[0] = 0;
    logdet[0] = 0;
    for (int i = 0; i < n; i++) {
        
        ySy[0] += l_ySy[i];
        logdet[0] += l_logdet[i];
        
        for (int j = 0; j < p; j++) {
            if (i == 0) {
                ySX[j] = 0;
            }
            
            ySX[j] += l_ySX[i * p + j];
            for (int k = 0; k < p; k++) {
                if (i == 0) {
                    XSX[j * p + k] = 0;
                }
                XSX[j * p + k] += l_XSX[i * p * p + j * p + k];
                for (int l = 0; l < nparms; l++) {
                    if (i == 0) {
                        dXSX[j * p * nparms + k * nparms + l] = 0;
                    }
                    dXSX[j * p * nparms + k * nparms + l] += l_dXSX[i * p * p * nparms + j * p * nparms + k * nparms + l];
                    //dXSX[j * p * nparms + k * nparms + l] += 0;
                }
            }
            for (int k = 0; k < nparms; k++) {
                if (i == 0) {
                    dySX[j * nparms + k] = 0;
                }
                dySX[j * nparms + k] += l_dySX[i * p * nparms + j * nparms + k];
                //printf("%f ", l_dySX[i * p * nparms + j * nparms + k]);
            }
        }
        for (int j = 0; j < nparms; j++) {
            if (i == 0) {
                dySy[j] = 0;
                dlogdet[j] = 0;
            }
            dySy[j] += l_dySy[i * nparms + j];
            dlogdet[j] += l_dlogdet[i * nparms + j];
            for (int k = 0; k < nparms; k++) {
                if (i == 0) {
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
    // printf("call_compute_pieces_fisher_gpu...done\n");
    
    
}

__global__ void call_Linv_mult(double* Linv, double* NNarray, double* z, double* x, int n, int m){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        int bsize = min(i+1,m);
        double temp = 0;
        for(int j=0; j<bsize; j++){
            temp += z[ static_cast<int>(NNarray[i * m + j]) - 1 ] * Linv[i * m + j];
        }
        x[i] = temp;
    }
}

extern "C"
double* call_Linv_mult_gpu(double* Linv, double* z, double* NNarray, int n, int m){
    double* d_Linv;
    double* d_NNarray;
    double* d_z;
    double* d_x;

    gpuErrchk(cudaMalloc((void**)&d_Linv, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_z, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_x, sizeof(double) * n));

    gpuErrchk(cudaMemcpy(d_Linv, Linv, sizeof(double) * n * m, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_z, z, sizeof(double) * n, cudaMemcpyHostToDevice));

    int grid_size = 64;
    int block_size = ((n + grid_size) / grid_size);

    call_Linv_mult << <block_size, grid_size >> > (d_Linv, d_NNarray, d_z, d_x, n, m);
    cudaDeviceSynchronize();

    double* x = (double*)malloc(sizeof(double) * n);
    gpuErrchk(cudaMemcpy(x, d_x, sizeof(double) * n, cudaMemcpyDeviceToHost));

    return x;
}
