#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
//' GPU Error check function
//`
//' Kernels do not throw exceptions. They instead return exit codes. If the exit code is
//` not \code{cudaSuccess} an error message is printed and the code is aborted.
#define gpuErrchk2(ans) { gpuAssert((ans), __FILE__, __LINE__); }
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

#define INDEX(i, j) (i * (i + 1) / 2 + j)

// locs n * dim
// dist array of size n * (n + 1) / 2 representing a triangular matrix
__global__ void calculate_distance_matrix(float* locs, float* dist, int* indicies, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < j || i > n) {
        return;
    }
    float d = 0;
    float temp = 0;
    for (int k = 0; k < dim; k++) {
        temp = (locs[i * dim + k] - locs[j * dim + k]);
        d += temp * temp;
    }
    dist[INDEX(i, j)] = d;
    indicies[INDEX(i, j)] = j + 1;
}

__device__ int partition(float* dist, int* indicies, int left, int right, int row, int pivotIndex) {
    float pivotValue = dist[INDEX(row, pivotIndex)];
    dist[INDEX(row, pivotIndex)] = dist[INDEX(row, right)];
    dist[INDEX(row, right)] = pivotValue;
	
	int temp3 = indicies[INDEX(row, pivotIndex)];	
	indicies[INDEX(row, pivotIndex)] = indicies[INDEX(row, right)];
    indicies[INDEX(row, right)] = temp3;

	int storeIndex = left;
    for (int i = left; i < right; i++){
        if (dist[INDEX(row, i)] < pivotValue) {
            float temp = dist[INDEX(row, i)];
            dist[INDEX(row, i)] = dist[INDEX(row, storeIndex)];
            dist[INDEX(row, storeIndex)] = temp;

            int temp2 = indicies[INDEX(row, i)];
            indicies[INDEX(row, i)] = indicies[INDEX(row, storeIndex)];
            indicies[INDEX(row, storeIndex)] = temp2;
			storeIndex++;
        }
    }
    float temp = dist[INDEX(row, right)];
    dist[INDEX(row, right)] = dist[INDEX(row, storeIndex)];
    dist[INDEX(row, storeIndex)] = temp;

    int temp2 = indicies[INDEX(row, right)];
    indicies[INDEX(row, right)] = indicies[INDEX(row, storeIndex)];
    indicies[INDEX(row, storeIndex)] = temp2;
    return storeIndex;
}

__device__ int hoare_partition(float* dist, int* indicies, int lo, int hi, int row, int pivotIndex) {
  float pivot = dist[INDEX(row, (hi - lo)/2 + lo)];
  int i = lo - 1;
  int j = hi + 1;
  while (i < j) {
    do {
      i++;
    } while (dist[INDEX(row, i)] < pivot);
    do {
      j--;
    }
    while (dist[INDEX(row, j)] > pivot);

    if (i >= j){
      return j;
    }
    float temp = dist[INDEX(row, i)];
    dist[INDEX(row, i)] = dist[INDEX(row, j)];
    dist[INDEX(row, j)] = temp;

    int temp2 = indicies[INDEX(row, i)];
    indicies[INDEX(row, i)] = indicies[INDEX(row, j)];
    indicies[INDEX(row, j)] = temp2;
  }
  return -1;
}
__device__ void select(float* dist, int* indicies, int row, int left, int right, int k){
    int pivotIndex;
    while (left < right) {
        pivotIndex = (left + right) / 2;
        pivotIndex = hoare_partition(dist, indicies, left, right, row, pivotIndex);
        if (k == pivotIndex) {
            return;
        } else if (k < pivotIndex) {
            right = pivotIndex;
        } else {
            left = pivotIndex + 1;
        }
    }
}

__global__ void sort_distance_matrix(float* dist, int* indicies, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || i < m + 1) {
        return;
    }
    select(dist, indicies, i, 0, i, m);
}

__global__ void create_nn_array(int* indicies, int* NNarray, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j > m || i < j){
        return;
    }
    NNarray[i * (m + 1) + j] = indicies[INDEX(i, j)];
}







__global__ void create_nn_array2(float* indicies, float* NNarray, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j > m || i < j){
        return;
    }
    NNarray[i * (m + 1) + j] = indicies[INDEX(i, j)];
}

#define INDEX_TRAP(i, j, h, nq) (i * h * nq + i * (i + 1) / 2 + j)

__device__ int hoare_partition_trap(float* dist, int* indicies, int lo, int hi, int row, int pivotIndex, int h, int nq) {
  float pivot = dist[INDEX_TRAP(row, (hi - lo)/2 + lo, h, nq)];
  int i = lo - 1;
  int j = hi + 1;
  while (i < j) {
    do {
      i++;
    } while (dist[INDEX_TRAP(row, i, h, nq)] < pivot);
    do {
      j--;
    }
    while (dist[INDEX_TRAP(row, j, h, nq)] > pivot);

    if (i >= j){
      return j;
    }
    float temp = dist[INDEX_TRAP(row, i, h, nq)];
    dist[INDEX_TRAP(row, i, h, nq)] = dist[INDEX_TRAP(row, j, h, nq)];
    dist[INDEX_TRAP(row, j, h, nq)] = temp;

    int temp2 = indicies[INDEX_TRAP(row, i, h, nq)];
    indicies[INDEX_TRAP(row, i, h, nq)] = indicies[INDEX_TRAP(row, j, h, nq)];
    indicies[INDEX_TRAP(row, j, h, nq)] = temp2;
  }
  return -1;
}
__device__ void select_trap(float* dist, int* indicies, int row, int left, int right, int k, int h, int nq){
    int pivotIndex;
    while (left < right) {
        pivotIndex = (left + right) / 2;
        pivotIndex = hoare_partition_trap(dist, indicies, left, right, row, pivotIndex, h, nq);
        if (k == pivotIndex) {
            return;
        } else if (k < pivotIndex) {
            right = pivotIndex;
        } else {
            left = pivotIndex + 1;
        }
    }
}

__global__ void nn_single(float* locs, float* dist, int* indicies, int* NNarray, int n, int m, int dim, int nq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nq) {
		return;
	}
    int index = i;
    int h = 0;
    while (index < n) {
        for (int j = 0; j <= index; j++){
            float d = 0;
            float temp = 0;
            for (int k = 0; k < dim; k++) {
                temp = (locs[index * dim + k] - locs[j * dim + k]);
                d += temp * temp;
            }
            dist[INDEX_TRAP(i, j, h, nq)] = d;
            indicies[INDEX_TRAP(i, j, h, nq)] = j + 1;
        }

        select_trap(dist, indicies, i, 0, index, m, h, nq);

        for (int j = 0; j < m + 1; j++) {
            NNarray[index * (m + 1) + j] = indicies[INDEX_TRAP(i, j, h, nq)];
        }

        index += nq; //nq is number of threads
        h++;
    }
}
extern "C"
int* nearest_neighbors_sing(float* locs, int m, int n, int dim, int nq) {

    float *d_locs, *d_dist;
	int *d_NNarray;
	int* d_indicies;

    gpuErrchk2(cudaMalloc((void**)&d_locs, sizeof(float) * n * dim));
    gpuErrchk2(cudaMalloc((void**)&d_NNarray, sizeof(int) * n * (m + 1)));
    gpuErrchk2(cudaMemcpy(d_locs, locs, sizeof(float) * n * dim, cudaMemcpyHostToDevice));
    
    
    gpuErrchk2(cudaMalloc((void**)&d_indicies, sizeof(int) * nq * (n - nq) + nq * (nq + 1) / 2));
    gpuErrchk2(cudaMalloc((void**)&d_dist, sizeof(float) * nq * (n - nq) + nq * (nq + 1) / 2));

    // dim3 threadsPerBlock2(32,1);
    // dim3 numBlocks2((n + 32 - 1) / 32, 1);
    int threadsPerBlock2 = 32;
    int numBlocks2 = (nq + 32 - 1) / 32;
    nn_single<<<numBlocks2, threadsPerBlock2>>>(d_locs, d_dist, d_indicies, d_NNarray, n, m, dim, nq);
    cudaDeviceSynchronize();
    
    int* NNarray = (int*) malloc(sizeof(int) * n * (m + 1));
    gpuErrchk2(cudaMemcpy(NNarray, d_NNarray, sizeof(int) * n * (m + 1), cudaMemcpyDeviceToHost));
    
    // float* NNarray = (float*) malloc(sizeof(float) * n * (m + 1));
    // gpuErrchk2(cudaMemcpy(NNarray, d_dist, sizeof(float) * n * (m + 1), cudaMemcpyDeviceToHost));
    
	return NNarray;
}


extern "C"
int* nearest_neighbors(float* locs, int m, int n, int dim) {

    float *d_locs, *d_dist;
	int *d_NNarray;
	//float *d_NNarray;
	int* d_indicies;

    gpuErrchk2(cudaMalloc((void**)&d_locs, sizeof(float) * n * dim));
    gpuErrchk2(cudaMalloc((void**)&d_NNarray, sizeof(int) * n * (m + 1)));
    gpuErrchk2(cudaMemcpy(d_locs, locs, sizeof(float) * n * dim, cudaMemcpyHostToDevice));
    
    
    gpuErrchk2(cudaMalloc((void**)&d_indicies, sizeof(int) * n * (n + 1) / 2));
    gpuErrchk2(cudaMalloc((void**)&d_dist, sizeof(float) * n * (n + 1) / 2));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x, (n+threadsPerBlock.y -1) / threadsPerBlock.y);
    calculate_distance_matrix<<<numBlocks,threadsPerBlock>>>(d_locs, d_dist, d_indicies, n, dim);
    cudaDeviceSynchronize();

    // dim3 threadsPerBlock2(32,1);
    // dim3 numBlocks2((n + 32 - 1) / 32, 1);
    int threadsPerBlock2 = 32;
    int numBlocks2 = (n + 32 - 1) / 32;
    sort_distance_matrix<<<numBlocks2, threadsPerBlock2>>>(d_dist, d_indicies, n, m);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock3(32, m + 1);
    create_nn_array<<<numBlocks2 , threadsPerBlock3>>>(d_indicies, d_NNarray, n, m);
    // create_nn_array2<<<numBlocks2 , threadsPerBlock3>>>(d_dist, d_NNarray, n, m);
    cudaDeviceSynchronize();
    
    int* NNarray = (int*) malloc(sizeof(int) * n * (m + 1));
    gpuErrchk2(cudaMemcpy(NNarray, d_NNarray, sizeof(int) * n * (m + 1), cudaMemcpyDeviceToHost));
    
    // float* NNarray = (float*) malloc(sizeof(float) * n * (m + 1));
    // gpuErrchk2(cudaMemcpy(NNarray, d_dist, sizeof(float) * n * (m + 1), cudaMemcpyDeviceToHost));
    
	return NNarray;
}
