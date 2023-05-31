#include <Rcpp.h>
using namespace Rcpp;

extern "C"
double* vecchia_Linv_gpu_outer(
    double* covparms,
    double* locs,
    double* NNarray,
    int n,
    int m,
    int dim);

// [[Rcpp::export]]
double meanC(NumericVector x) {
  int n = x.size();
  double total = 0;

  for (int i = 0; i < n; ++i) {
    total += x[i];
  }
  return total / n;
}

// [[Rcpp::export]]
NumericMatrix vecchia_Linv_gpu_isotropic_exponential(
    NumericVector covparms,
    NumericMatrix locs,
    NumericMatrix NNarray) {
    
    int m = NNarray.ncol();
    int n = locs.nrow();
    int nparms = covparms.length();
    int dim = locs.ncol();

    double covparmsl[3] = { 0, 0, 0 };
    covparmsl[0] = covparms[0];
    covparmsl[1] = covparms[1];
    covparmsl[2] = covparms[2];

    double* locsl = (double*)malloc(sizeof(double) * n * dim);
    double* NNarrayl = (double*)malloc(sizeof(double) * n * m);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            if (j < dim) {
                locsl[i * dim + j] = locs(i, j);
            }
            NNarrayl[i * m + j] = NNarray(i, j);
        }
    }

    double* Linvl = vecchia_Linv_gpu_outer(covparmsl, locsl, NNarrayl, n, m, dim);

    NumericMatrix Linv( n , m );

    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            Linv(i,j) = Linvl[i * m + j];
        }
    }
    return Linv;
}