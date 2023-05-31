// #include <Rcpp.h>
#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

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
arma::mat vecchia_Linv_gpu_isotropic_exponential(
    arma::vec covparms,
    arma::mat locs,
    arma::mat NNarray) {
    
    int m = NNarray.n_cols;
    int n = locs.n_rows;
    int nparms = covparms.n_elem;
    int dim = locs.n_cols;

    double covparmsl[3] = { 0, 0, 0 };
    covparmsl[0] = covparms[0];
    covparmsl[1] = covparms[1];
    covparmsl[2] = covparms[2];

    // double* locsl = (double*)malloc(sizeof(double) * n * dim);
    locs = locs.t();
    double* locsl = locs.memptr();
    // double* NNarrayl = (double*)malloc(sizeof(double) * n * m);
    NNarray = NNarray.t();
    double* NNarrayl = NNarray.memptr();

    // for (int i = 0; i < n; i++){
    //     for (int j = 0; j < m; j++){
    //         if (j < dim) {
    //             // locsl[i * dim + j] = locs(i, j);
    //         }
    //         NNarrayl[i * m + j] = NNarray(i, j);
    //     }
    // }

    double* Linvl = vecchia_Linv_gpu_outer(covparmsl, locsl, NNarrayl, n, m, dim);

    // NumericMatrix Linv( n , m );
    // arma::mat Linv = arma::mat(n, m);
    // for (int i = 0; i < n; i++){
    //     for (int j = 0; j < m; j++){
    //         Linv(i,j) = Linvl[i * m + j];
    //     }
    // }
    arma::mat Linv = arma::mat(&Linvl[0], m, n, false);
    return Linv.t();
}