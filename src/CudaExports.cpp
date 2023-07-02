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

extern "C"
void call_compute_pieces_gpu(
    double* covparms,
    // std::string covfun_name,
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
);

// [[Rcpp::export]]
double exponential_isotropic_likelihood(
    NumericVector covparms,
    NumericMatrix locs,
    NumericMatrix NNarray,
    NumericVector y,
    NumericMatrix X,
    int profbeta,
    int grad_info
) {
    int n = y.length();
    int m = NNarray.ncol();
    int p = X.ncol();
    int nparms = covparms.length();
    int dim = locs.ncol();

    double covparmsl[3] = { 0, 0, 0 };
    covparmsl[0] = covparms[0];
    covparmsl[1] = covparms[1];
    covparmsl[2] = covparms[2];

    double* locsl = (double*)malloc(sizeof(double) * n * dim);
    double* NNarrayl = (double*)malloc(sizeof(double) * n * m);
    double* yl = (double*)malloc(sizeof(double) * n);
    double* Xl = (double*)malloc(sizeof(double) * n * p);

    for (int i = 0; i < n; i++) {
        yl[i] = y[i];
        for (int j = 0; j < m; j++) {
            if (j < dim) {
                locsl[i * dim + j] = locs(i, j);
            }
            if (j < p) {
                Xl[i * p + j] = X(i, j);
            }
            NNarrayl[i * m + j] = NNarray(i, j);
        }
    }


    double* XSXf = (double*)calloc(p * p, sizeof(double));
    double* ySXf = (double*)calloc(p * p, sizeof(double));
    double ySy = 0;
    double logdet = 0;

    double* dXSXf = (double*)calloc(p * p * nparms, sizeof(double));
    double* dySXf = (double*)calloc(p * nparms, sizeof(double));
    double* dySyf = (double*)malloc(sizeof(double) * nparms);
    double* dlogdetf = (double*)malloc(sizeof(double) * nparms);
    double* ainfof = (double*)malloc(sizeof(double) * nparms * nparms);

    call_compute_pieces_gpu(covparmsl, /*covfun_name,*/ locsl, NNarrayl, yl, Xl,
        XSXf, ySXf, &ySy, &logdet,
        dXSXf, dySXf, dySyf, dlogdetf, ainfof, profbeta, grad_info, n, m, p, nparms, dim);

    double* fabeta = (double*)calloc(sizeof(double), p);
    double* fbetahat = (double*)calloc(sizeof(double), p);
    // if (profbeta) {
    // 	gauss_eliminate(XSXf, ySXf, fabeta, p);
    // }
    for (int j = 0; j < p; j++) { fbetahat[j] = fabeta[j]; };
    /*double sig2 = (ySy - 2.0 * as_scalar(ySX.t() * abeta) +
        as_scalar(abeta.t() * XSX * abeta)) / n;*/
    double fsig2 = ySy;
    double temp = 0;
    for (int i = 0; i < p; i++) {
        temp += ySXf[i] * fabeta[i];
    }
    //printf("temp: %f\n", temp);
    fsig2 -= 2.0 * temp;
    temp = 0;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            temp += XSXf[i * p + j] * fabeta[i] * fabeta[j];
        }
    }
    fsig2 += temp;
    fsig2 /= n;
    double fll = -0.5 * (n * std::log(2.0 * M_PI) + logdet + n * fsig2);
    // List ret = List::create( Named("loglik") = fll );
    // return ret;
    return fll;
}