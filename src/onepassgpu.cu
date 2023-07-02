

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
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

__global__ void compute_pieces(double* y, double* X, double* NNarray, double* locs, double* locsub,
    double* covmat, double* logdet, double* ySy, double* XSX, double* ySX,
    double* dXSX, double* dySX, double* dySy, double* dlogdet, double* ainfo,
    double variance, double range, double nugget,
    int n, int p, int m, int dim, int nparms,
    bool profbeta, bool grad_info,
    double* dcovmat,
    double* ysub, double* X0, double* Liy0, double* LiX0, double* choli2, double* onevec,
    double* LidSLi2, double* c, double* v1, double* LidSLi3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int bsize = std::min(i + 1, m);
    clock_t full_start = clock();
    if (i >= m && i < n) {
        clock_t start = clock();
        for (int j = m - 1; j >= 0; j--) {
            ysub[i * m + m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
            for (int k = 0; k < dim; k++) {
                locsub[i * m * dim + (m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
            }
            if (profbeta) {
                for (int k = 0; k < p; k++) {
                    X0[i * m * p + (m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k];
                }
            }
        }
        /*if (i == 40) {
            printf("GPU X0\n");
            for (int j = 0; j < 4; j++) {

                printf("%f ", X0[40 * m * p + j * p + 0]);
            }
        }*/
        clock_t end = clock();
        if (i == 10000) {
            printf("Substituting NN + scaling: %i\n", end - start);
        }
        start = clock();
        // compute covariance matrix and derivatives and take cholesky
        // arma::mat covmat = p_covfun[0](covparms, locsub);
        //double* covmat = exponential_isotropic(covparms, locsub, m, dim);
        // Calculate covmatrix
        double temp;
        for (int i1 = 0; i1 < m; i1++) {
            for (int i2 = 0; i2 <= i1; i2++) {
                double d = 0.0;
                for (int j = 0; j < dim; j++) {
                    temp = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                    d += temp * temp;
                }
                d = sqrt(d);
                // calculate covariance
                if (i1 == i2) {
                    covmat[i * m * m + i2 * m + i1] = variance * (exp(-d) + nugget);
                }
                else {
                    covmat[i * m * m + i2 * m + i1] = variance * exp(-d);
                    covmat[i * m * m + i1 * m + i2] = covmat[i * m * m + i2 * m + i1];
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
        if (i == 10000) {
            printf("Covariance matrix: %i\n", end - start);
        }

        start = clock();
        if (grad_info) {
            // calculate derivatives
            //arma::cube dcovmat = arma::cube(n, n, covparms.n_elem, fill::zeros);
            //dcovmat = (double*)malloc(sizeof(double) * m * m * nparms);
            for (int i1 = 0; i1 < m; i1++) {
                for (int i2 = 0; i2 <= i1; i2++) {
                    double d = 0.0;
                    double a = 0;
                    for (int j = 0; j < dim; j++) {
                        a = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                        d += a * a;
                    }
                    d = sqrt(d);
                    temp = exp(-d);

                    dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += temp;
                    dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 1] += variance * temp * d / range;
                    if (i1 == i2) { // update diagonal entry
                        dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += nugget;
                        dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 2] = variance;
                    }
                    else { // fill in opposite entry
                        for (int j = 0; j < nparms; j++) {
                            dcovmat[i * m * m * nparms + i2 * m * nparms + i1 * nparms + j] = dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + j];
                        }
                    }
                }
            }

        }
        end = clock();
        if (i == 10000) {
            printf("Covariance derivative: %i\n", end - start);
        }
        start = clock();
        /*arma::mat cholmat = eye(size(covmat));
        chol(cholmat, covmat, "lower");*/

        // Cholesky decomposition
        //int k, q, j;
        double temp2;
        double diff;
        /*for (int q = 0; q < m; q++) {
            diff = 0;
            temp2 = covmat[i * m * m + q * m + q];
            for (int k = 0; k < m; k++) {
                if (k < q) {
                    temp = covmat[i * m * m + q * m + k];
                    diff -= temp * temp;
                }
                else if (k == q) {
                    covmat[i * m * m + q * m + q] = sqrt(covmat[i * m * m + q * m + q] + diff);
                    diff = 0;
                }
                else {
                    diff = 0;
                    for (int r = 0; r < q; r++) {
                        diff += covmat[i * m * m + q * m + r] * covmat[i * m * m + k * m + r];
                    }
                    covmat[i * m * m + k * m + q] = (covmat[i * m * m + k * m + q] - diff) / temp2;
                }
            }
        }*/
        int r, j, k, l;
        int retval = 1;
        for (r = 0; r < m + 0; r++) {
            diff = 0;
            for (k = 0; k < r; k++) {
                temp = covmat[i * m * m + r * m + k];
                diff += temp * temp;
            }
            covmat[i * m * m + r * m + r] = sqrt(covmat[i * m * m + r * m + r] - diff);


            for (j = r + 1; j < m + 0; j++) {
                diff = 0;
                for (k = 0; k < r; k++) {
                    diff += covmat[i * m * m + r * m + k] * covmat[i * m * m + j * m + k];
                }
                covmat[i * m * m + j * m + r] = (covmat[i * m * m + j * m + r] - diff) / covmat[i * m * m + r * m + r];
            }
        }

        end = clock();
        if (i == 10000) {
            printf("Cholesky decomposition: %i\n", end - start);
        }


        start = clock();
        // i1 is conditioning set, i2 is response        
        //arma::span i1 = span(0,bsize-2);
        //arma::span i2 = span(bsize - 1, bsize - 1);

        // get last row of cholmat
        /*arma::vec onevec = zeros(bsize);
        onevec(bsize - 1) = 1.0;*/

        if (grad_info) {
            //choli2 = backward_solve(cholmat, onevec, m);
            choli2[i * m + m - 1] = 1 / covmat[i * m * m + (m - 1) * m + m - 1];

            for (int k = m - 2; k >= 0; k--) {
                double dd = 0.0;
                for (int j = m - 1; j > k; j--) {
                    dd += covmat[i * m * m + j * m + k] * choli2[i * m + j];
                }
                choli2[i * m + k] = (-dd) / covmat[i * m * m + k * m + k];
            }
        }
        end = clock();
        if (i == 10000) {
            printf("solve(cholmat, onevec, m): %i\n", end - start);
        }
        //bool cond = bsize > 1;

        // do solves with X and y
        if (profbeta) {
            start = clock();
            // LiX0 = forward_solve_mat(cholmat, X0, m, p);
            for (int k = 0; k < p; k++) {
                LiX0[i * m * p + 0 * p + k] = X0[i * m * p + 0 * p + k] / covmat[i * m * m + 0 * m + 0];
            }

            for (int h = 1; h < m; h++) {
                for (int k = 0; k < p; k++) {
                    double dd = 0.0;
                    for (int j = 0; j < h; j++) {
                        dd += covmat[i * m * m + h * m + j] * LiX0[i * m * p + j * p + k];
                    }
                    LiX0[i * m * p + h * p + k] = (X0[i * m * p + h * p + k] - dd) / covmat[i * m * m + h * m + h];
                }
            }
            end = clock();
            if (i == 10000) {
                printf("forward_solve_mat(cholmat, X0, m, p): %i\n", end - start);
            }

        }
        for (int j = 0; j < m; j++) {
            for (int k = j + 1; k < m; k++) {
                covmat[i * m * m + j * m + k] = 0.0;
            }
        }

        //arma::vec Liy0 = solve( trimatl(cholmat), ysub );
        //double* Liy0 = forward_solve(cholmat, ysub, m);
        //double* Liy0 = (double*)malloc(sizeof(double) * m);
        /*for (int j = 0; j < m; j++) {
            Liy0[i * m + j] = 0.0f;
        }*/
        start = clock();
        Liy0[i * m + 0] = ysub[i * m + 0] / covmat[i * m * m + 0 * m + 0];

        for (int k = 1; k < m; k++) {
            double dd = 0.0;
            for (int j = 0; j < k; j++) {
                dd += covmat[i * m * m + k * m + j] * Liy0[i * m + j];
            }
            Liy0[i * m + k] = (ysub[i * m + k] - dd) / covmat[i * m * m + k * m + k];
        }
        end = clock();
        if (i == 10000) {
            printf("forward_solve(cholmat, ysub, m): %i\n", end - start);
            start = clock();
        }

        // loglik objects
        logdet[i] = 2.0 * log(covmat[i * m * m + (m - 1) * m + m - 1]);

        temp = Liy0[i * m + m - 1];
        ySy[i] = temp * temp;


        if (profbeta) {
            start = clock();
            /*l_XSX += LiX0.rows(i2).t() * LiX0.rows(i2);
            l_ySX += (Liy0(i2) * LiX0.rows(i2)).t();*/
            temp2 = Liy0[i * m + m - 1];
            for (int i1 = 0; i1 < p; i1++) {
                temp = LiX0[i * m * p + (m - 1) * p + i1];
                for (int i2 = 0; i2 <= i1; i2++) {
                    XSX[i * p * p + i1 * p + i2] = temp * LiX0[i * m * p + (m - 1) * p + i2];
                    XSX[i * p * p + i2 * p + i1] = XSX[i * p * p + i1 * p + i2];
                }
                ySX[i * p + i1] = temp2 * LiX0[i * m * p + (m - 1) * p + i1];
            }
            end = clock();
            if (i == 10000) {
                printf("Liy0(i2) * LiX0.rows(i2): %i\n", end - start);
                start = clock();
            }
        }
        if (grad_info) {
            // gradient objects
            // LidSLi3 is last column of Li * (dS_j) * Lit for 1 parameter i
            // LidSLi2 stores these columns in a matrix for all parameters
            // arma::mat LidSLi2(bsize, nparms);



            start = clock();
            for (int j = 0; j < nparms; j++) {
                // compute last column of Li * (dS_j) * Lit
                //arma::vec LidSLi3 = forward_solve(cholmat, dcovmat.slice(j) * choli2);
                // c = dcovmat.slice(j) * choli2
                for (int h = 0; h < m; h++) {
                    c[i * m + h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        temp += dcovmat[i * m * m * nparms + h * m * nparms + k * nparms + j] * choli2[i * m + k];
                    }
                    c[i * m + h] = temp;
                }

                //LidSLi3 = forward_solve(cholmat, c);      
                LidSLi3[i * m + 0] = c[i * m + 0] / covmat[i * m * m + 0 * m + 0];

                for (int k = 1; k < m; k++) {
                    double dd = 0.0;
                    for (int l = 0; l < k; l++) {
                        dd += covmat[i * m * m + k * m + l] * LidSLi3[i * m + l];
                    }
                    LidSLi3[i * m + k] = (c[i * m + k] - dd) / covmat[i * m * m + k * m + k];
                }

                ////////////////
                //arma::vec v1 = LiX0.t() * LidSLi3;

                for (int h = 0; h < p; h++) {
                    v1[i * p + h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        temp += LiX0[i * m * p + k * p + h] * LidSLi3[i * m + k];
                    }
                    v1[i * p + h] = temp;
                }

                ////////////////

                //double s1 = as_scalar(Liy0.t() * LidSLi3);
                double s1 = 0;
                for (int h = 0; h < m; h++) {
                    s1 += Liy0[i * m + h] * LidSLi3[i * m + h];
                }

                ////////////////

                /*(l_dXSX).slice(j) += v1 * LiX0.rows(i2) + (v1 * LiX0.rows(i2)).t() -
                    as_scalar(LidSLi3(i2)) * (LiX0.rows(i2).t() * LiX0.rows(i2));*/

                    //double* v1LiX0 = (double*)malloc(sizeof(double) * m * m);
                double temp3;
                double temp4 = LidSLi3[i * m + m - 1];
                for (int h = 0; h < p; h++) {
                    temp = v1[i * p + h];
                    temp2 = LiX0[i * m * p + (m - 1) * p + h];

                    for (int k = 0; k < p; k++) {
                        temp3 = LiX0[i * m * p + (m - 1) * p + k];
                        dXSX[i * p * p * nparms + h * p * nparms + k * nparms + j] = temp * temp3 +
                            (v1[i * p + k] - temp4 * temp3) * temp2;
                    }
                }
                temp = Liy0[i * m + m - 1];
                ///////////////
                /*(l_dySy)(j) += as_scalar(2.0 * s1 * Liy0(i2) -
                    LidSLi3(i2) * Liy0(i2) * Liy0(i2));*/
                dySy[i * nparms + j] = (2 * s1 - temp4 * temp) * temp;

                /*(l_dySX).col(j) += (s1 * LiX0.rows(i2) + (v1 * Liy0(i2)).t() -
                    as_scalar(LidSLi3(i2)) * LiX0.rows(i2) * as_scalar(Liy0(i2))).t();*/
                temp3 = LidSLi3[i * m + m - 1];
                for (int h = 0; h < p; h++) {
                    temp2 = LiX0[i * m * p + (m - 1) * p + h];
                    dySX[i * p * nparms + h * nparms + j] = s1 * temp2 +
                        v1[i * p + h] * temp - temp3 * temp2 * temp;
                }

                //(l_dlogdet)(j) += as_scalar(LidSLi3(i2));
                dlogdet[i * nparms + j] = temp3;

                //LidSLi2.col(j) = LidSLi3;
                for (int h = 0; h < m; h++) {
                    LidSLi2[i * m * nparms + h * nparms + j] = LidSLi3[i * m + h];
                }
                /*if (i == 40 && j == 2) {
                    printf("CPU s1\n");
                    printf("%f", s1);
                }*/

            }
            end = clock();
            if (i == 10000) {
                printf("gradient objects: %i\n", end - start);
                start = clock();
            }
            start = clock();
            // fisher information object
            // bottom right corner gets double counted, so subtract it off
            for (int h = 0; h < nparms; h++) {
                temp2 = LidSLi2[i * m * nparms + (m - 1) * nparms + h];
                for (int j = 0; j < h + 1; j++) {
                    /*(l_ainfo)(h, j) +=
                        1.0 * accu(LidSLi2.col(h) % LidSLi2.col(j)) -
                        0.5 * accu(LidSLi2.rows(i2).col(j) %
                            LidSLi2.rows(i2).col(h));*/
                    double s = 0;
                    for (int l = 0; l < m; l++) {
                        s += LidSLi2[i * m * nparms + l * m + h] * LidSLi2[i * m * nparms + l * m + j];
                    }
                    ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[i * m * nparms + (m - 1) * nparms + j] * temp2;
                }
            }
            end = clock();
            if (i == 10000) {
                printf("fisher information object: %i\n", end - start);
                start = clock();
            }
        }
    }
    clock_t full_end = clock();
    if (i == 10000) {
        printf("Total time: %i\n", full_end - full_start);
    }
}


extern "C"
void call_compute_pieces_gpu(
    double* covparms,
    std::string covfun_name,
    double* locs,
    double* NNarray,
    double* y,
    double* X,
    double* XSX,
    double* ySX,
    double* ySy,
    double* logdet,
    double* dXSX,
    double* dySX,
    double* dySy,
    double* dlogdet,
    double* ainfo,
    int profbeta,
    int grad_info,
    int n,
    int m,
    int p,
    int nparms,
    int dim
) {
    //m++;

    double* d_locs;
    double* d_NNarray;
    double* d_y;
    double* d_X;

    gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
    gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_y, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_X, sizeof(double) * n * p));

    double* d_covmat;
    double* d_locs_scaled;
    double* d_ySX;
    double* d_XSX;
    double* d_ySy;
    double* d_logdet;
    double* d_dXSX;
    double* d_dySX;
    double* d_dySy;
    double* d_dlogdet;
    double* d_dainfo;

    double* d_dcovmat;
    double* d_ysub;
    double* d_X0;
    double* d_Liy0;
    double* d_LiX0;
    double* d_choli2;
    double* d_onevec;
    double* d_LidSLi2;
    double* d_c;
    double* d_v1;
    double* d_LidSLi3;

    gpuErrchk(cudaMalloc((void**)&d_covmat, sizeof(double) * n * m * m));
    gpuErrchk(cudaMalloc((void**)&d_locs_scaled, sizeof(double) * n * m * dim));
    gpuErrchk(cudaMalloc((void**)&d_ySX, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_XSX, sizeof(double) * n * p * p));
    gpuErrchk(cudaMalloc((void**)&d_ySy, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_logdet, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_dXSX, sizeof(double) * n * p * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySX, sizeof(double) * n * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySy, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dlogdet, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dainfo, sizeof(double) * n * nparms * nparms));

    gpuErrchk(cudaMalloc((void**)&d_dcovmat, sizeof(double) * n * m * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_ysub, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_X0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_Liy0, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_LiX0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_choli2, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_onevec, sizeof(double) * m));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi2, sizeof(double) * n * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_c, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_v1, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi3, sizeof(double) * n * m));

    gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_X, X, sizeof(double) * n * p, cudaMemcpyHostToDevice));

    int grid_size = 64;
    int block_size = ((n + grid_size) / grid_size);

    auto begin = std::chrono::steady_clock::now();
    compute_pieces << <block_size, grid_size >> > (d_y, d_X, d_NNarray, d_locs, d_locs_scaled,
        d_covmat, d_logdet, d_ySy, d_XSX, d_ySX,
        d_dXSX, d_dySX, d_dySy, d_dlogdet, d_dainfo,
        covparms[0], covparms[1], covparms[2],
        n, p, m, dim, nparms,
        profbeta, grad_info,
        d_dcovmat,
        d_ysub, d_X0, d_Liy0, d_LiX0, d_choli2, d_onevec,
        d_LidSLi2, d_c, d_v1, d_LidSLi3);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    std::cout << "Kernel difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[mus]" << std::endl;


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

    /// Testing
    /*double* l_Liy0 = (double*)malloc(sizeof(double) * n * m);
    gpuErrchk(cudaMemcpy(l_Liy0, d_Liy0, sizeof(double) * n * m, cudaMemcpyDeviceToHost));

    printf("CPU Liy0\n");
    for (int j = 0; j < 10; j++) {
        printf("%f ", l_Liy0[40 * m + j]);
    }
    printf("\n");*/
    //////////////////



    ySy[0] = 0;
    logdet[0] = 0;
    for (int i = 0; i < n; i++) {
        /*if (i < 50) {
            printf("%f\n", l_ySy[i]);
        }*/
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
        //printf("%f\n", l_logdet[i]);
    }
    //printf("m:%i\n", m);
    //arma::vec covparmsa(covparms, 3);
    //arma::mat locsa(locs, dim, n);
    //arma::mat NNarraya(NNarray, m, m);
    //arma::vec ya(y, m);
    //arma::mat Xa(X, p, m);

    //locsa = locsa.t();
    //NNarraya = NNarraya.t();
    //Xa = Xa.t();

    //arma::mat XSXa = arma::mat(p, p, fill::zeros);
    //arma::vec ySXa = arma::vec(p, fill::zeros);
    //double ySya = 0.0;
    //double logdeta = 0.0;

    //// gradient objects    
    //arma::cube dXSXa = arma::cube(p, p, nparms, fill::zeros);
    //arma::mat dySXa = arma::mat(p, nparms, fill::zeros);
    //arma::vec dySya = arma::vec(nparms, fill::zeros);
    //arma::vec dlogdeta = arma::vec(nparms, fill::zeros);
    //// fisher information
    //arma::mat ainfoa = arma::mat(nparms, nparms, fill::zeros);

    //try {
    //    compute_pieces(
    //        covparmsa, "exponential_isotropic", locsa, NNarraya, ya, Xa,
    //        &XSXa, &ySXa, &ySya, &logdeta, &dXSXa, &dySXa, &dySya, &dlogdeta, &ainfoa,
    //        profbeta, grad_info
    //    );
    //}
    //catch (std::exception& exc) {
    //    printf(exc.what());
    //}
    //printf("logdet: %f\n", logdet[0]);
    //    
    //ySy[0] += ySya;
    //logdet[0] += logdeta;
    //for (int j = 0; j < p; j++) {
    //    ySX[j] += ySXa(j);
    //    for (int k = 0; k < p; k++) {
    //        XSX[j * p + k] += XSXa(j, k);
    //        for (int l = 0; l < nparms; l++) {
    //            dXSX[j * p * nparms + k * nparms + l] += dXSXa(j, k, l);
    //            //dXSX[j * p * nparms + k * nparms + l] += 0;
    //        }
    //    }
    //    for (int k = 0; k < nparms; k++) {
    //        dySX[j * nparms + k] += dySXa(j, k);
    //        //printf("%f ", l_dySX[i * p * nparms + j * nparms + k]);
    //    }
    //}
    //printf("%f\n", l_logdet[i]);
    /*l_ySy = (double*)malloc(sizeof(double) * n);
    l_logdet = (double*)malloc(sizeof(double) * n);*/
    l_ySX = (double*)calloc(p, sizeof(double));
    l_XSX = (double*)calloc(p * p, sizeof(double));
    l_dySX = (double*)calloc(p * nparms, sizeof(double));
    l_dXSX = (double*)calloc(p * p * nparms, sizeof(double));
    l_dySy = (double*)calloc(nparms, sizeof(double));
    l_dlogdet = (double*)calloc(nparms, sizeof(double));
    l_ainfo = (double*)calloc(nparms * nparms, sizeof(double));

    double ySya = 0;
    double logdeta = 0;
    try {
        compute_pieces_cpu(covparms[0], covparms[1], covparms[2],
            locs, NNarray, y, X,
            l_XSX, l_ySX, ySya, logdeta, l_dXSX, l_dySX, l_dySy, l_dlogdet, l_ainfo,
            profbeta, grad_info,
            m, m, p, nparms, dim);
    }
    catch (std::exception& exc) {
        printf(exc.what());
    }
    //printf("All good\n");
    ySy[0] += ySya;
    logdet[0] += logdeta;
    for (int j = 0; j < p; j++) {
        ySX[j] += l_ySX[j];
        for (int k = 0; k < p; k++) {
            XSX[j * p + k] += l_XSX[j * m + k];
            for (int l = 0; l < nparms; l++) {
                dXSX[j * p * nparms + k * nparms + l] += l_dXSX[j * p * nparms + nparms * k + l];
                //dXSX[j * p * nparms + k * nparms + l] += 0;
            }
        }
        for (int k = 0; k < nparms; k++) {
            dySX[j * nparms + k] += l_dySX[j * nparms + k];
            //printf("%f ", l_dySX[i * p * nparms + j * nparms + k]);
        }
    }
}