

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
        // fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

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
    /*__shared__ double ls[16 * 21 * 2];
    __shared__ double cov[16 * 21 * 21];*/
    
    //clock_t start = clock();
    for (int j = 0; j < m; j++) {
        for (int k = 0; k < dim; k++) {
            locs_scaled[i * m * dim + (m - 1 - j) * dim + k] = ( locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] ) / range;
            //ls[threadIdx.x * m * dim + (m - 1 -j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
        }
    }
    //clock_t end = clock();
    //if (i == 21) {
    //    printf("scale %i\n", (int)(end - start));
    //}    // Calculate covariances
    //start = clock();
    for (int i1 = 0; i1 < m; i1++) {
        for (int i2 = 0; i2 <= i1; i2++) {
            // calculate distance
            double d = hypot(locs_scaled[i * m * dim + i1 * dim ]- locs_scaled[i * m * dim + i2 * dim], 
                locs_scaled[i * m * dim + i1 * dim + 1] - locs_scaled[i * m * dim + i2 * dim + 1]);
            /*double d = hypot(ls[threadIdx.x * m * dim + i1 * dim] - ls[threadIdx.x * m * dim + i2 * dim],
                ls[threadIdx.x * m * dim + i1 * dim + 1] - ls[threadIdx.x * m * dim + i2 * dim + 1]);*/
            // calculate covariance
            if (i1 == i2) {
                covmat[i * m * m + i2 * m + i1] = variance * (expf(-d) + nugget);
                //cov[threadIdx.x * m * m + i2 * m + i1] = variance * (exp(-d) + nugget);
            }
            else {
                covmat[i * m * m + i2 * m + i1] = variance * expf(-d);
                /*cov[threadIdx.x * m * m + i2 * m + i1] = variance * exp(-d);
                cov[threadIdx.x * m * m + i1 * m + i2] = cov[threadIdx.x * m * m + i2 * m + i1];*/
                covmat[i * m * m + i1 * m + i2] = covmat[i * m * m + i2 * m + i1];
            }
        }
    }
    /*end = clock();
    if (i == 21) {
        printf("covmat %i\n", (int)(end - start));
    }*/
    clock_t start = clock();
 
    // Cholesky decomposition
    int k, q, j;
    double temp, diff;
    for (q = 0; q < m; q++) {
        diff = 0;
        for (k = 0; k < m; k++) {
            if (k < q) {
                //temp = cov[threadIdx.x * m * m + k * m + q];
                temp = covmat[i * m * m + k * m + q];
                diff -= temp * temp;
            }
            else if (k == q) {
                //cov[threadIdx.x * m * m + q * m + q] = sqrt(cov[threadIdx.x * m * m + q * m + q] + diff);
                covmat[i * m * m + q * m + q] = sqrt(covmat[i * m * m + q * m + q] + diff);;
                diff = 0;
            }
            else {
                diff = 0;
                for (int p = 0; p < q; p++) {
                    diff += covmat[i * m * m + p * m + q] * covmat[i * m * m + p * m + k];
                    //diff += cov[threadIdx.x * m * m + p * m + q] * cov[threadIdx.x * m * m + p * m + k];
                }
                covmat[i * m * m + q * m + k] = (covmat[i * m * m + q * m + k] - diff ) / covmat[i * m * m + q * m + q];
                //cov[threadIdx.x * m * m + q * m + k] = (cov[threadIdx.x * m * m + q * m + k] - diff) / cov[threadIdx.x * m * m + q * m + q];
            }
        }

    }
    //assert(retval != 0);
    /*end = clock();
    if (i == 21) {
        printf("cholesky %i\n", (int)(end - start));
    }*/

    // Solve system
    //start = clock();
    int b = 1;
    for (int q = m - 1; q >= 0; q--) {
        double sum = 0.0;
        for (int j = q + 1; j < m; j++) {
            sum += covmat[i * m * m + q * m + j] * NNarray[i * m + m - 1 - j];
            //sum += cov[threadIdx.x * m * m + q * m + j] * NNarray[i * m + m - 1 - j];
        }
        NNarray[i * m + m - 1 - q] = (b - sum) / covmat[i * m * m + q * m + q];
        //NNarray[i * m + m - 1 - q] = (b - sum) / cov[threadIdx.x * m * m + q * m + q];
        b = 0;
    }
    clock_t end = clock();
    /*if (i == 21) {
        printf("Cholesky + solve cycles %i\n", (int)(end - start));
    }*/
    /*if (i == 201) {
        for (int p = 0; p < m; p++) {
            printf("%f ", NNarray[i * m + p]);
        }
        printf("\n\n");
    }*/
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    

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
    gpuErrchk(cudaMalloc((void**)&d_covmat, sizeof(double) * n * m * m));
    gpuErrchk(cudaMalloc((void**)&d_locs_scaled, sizeof(double) * n * m * dim));

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
