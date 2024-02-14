// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// nearest_neighbors_gpu
NumericMatrix nearest_neighbors_gpu(arma::mat locs, int m);
RcppExport SEXP _GpGpU_nearest_neighbors_gpu(SEXP locsSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(nearest_neighbors_gpu(locs, m));
    return rcpp_result_gen;
END_RCPP
}
// nearest_neighbors_sing_gpu
NumericMatrix nearest_neighbors_sing_gpu(arma::mat locs, int m, int nq);
RcppExport SEXP _GpGpU_nearest_neighbors_sing_gpu(SEXP locsSEXP, SEXP mSEXP, SEXP nqSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type nq(nqSEXP);
    rcpp_result_gen = Rcpp::wrap(nearest_neighbors_sing_gpu(locs, m, nq));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_Linv_gpu
arma::mat vecchia_Linv_gpu(arma::vec covparms, StringVector covfun_name, arma::mat locs, arma::mat NNarray);
RcppExport SEXP _GpGpU_vecchia_Linv_gpu(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP locsSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_Linv_gpu(covparms, covfun_name, locs, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_Linv_gpu_isotropic_exponential_batched
arma::mat vecchia_Linv_gpu_isotropic_exponential_batched(arma::vec covparms, arma::mat locs, arma::mat NNarray);
RcppExport SEXP _GpGpU_vecchia_Linv_gpu_isotropic_exponential_batched(SEXP covparmsSEXP, SEXP locsSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_Linv_gpu_isotropic_exponential_batched(covparms, locs, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_profbeta_loglik_grad_info_gpu
List vecchia_profbeta_loglik_grad_info_gpu(NumericVector covparms, StringVector covfun_name, NumericVector y, NumericMatrix X, const NumericMatrix locs, NumericMatrix NNarray);
RcppExport SEXP _GpGpU_vecchia_profbeta_loglik_grad_info_gpu(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_profbeta_loglik_grad_info_gpu(covparms, covfun_name, y, X, locs, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_profbeta_loglik_gpu
List vecchia_profbeta_loglik_gpu(NumericVector covparms, StringVector covfun_name, NumericVector y, NumericMatrix X, const NumericMatrix locs, NumericMatrix NNarray);
RcppExport SEXP _GpGpU_vecchia_profbeta_loglik_gpu(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_profbeta_loglik_gpu(covparms, covfun_name, y, X, locs, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// fisher_scoring_gpu
List fisher_scoring_gpu(NumericVector start_params, NumericVector y, NumericMatrix X, NumericMatrix locs, NumericMatrix NNarray, double vv, bool silent, double convtol, int max_iter);
RcppExport SEXP _GpGpU_fisher_scoring_gpu(SEXP start_paramsSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNarraySEXP, SEXP vvSEXP, SEXP silentSEXP, SEXP convtolSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type start_params(start_paramsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type NNarray(NNarraySEXP);
    Rcpp::traits::input_parameter< double >::type vv(vvSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    Rcpp::traits::input_parameter< double >::type convtol(convtolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(fisher_scoring_gpu(start_params, y, X, locs, NNarray, vv, silent, convtol, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// Linv_mult_gpu
NumericVector Linv_mult_gpu(NumericMatrix Linv, NumericVector z, IntegerMatrix NNarray);
RcppExport SEXP _GpGpU_Linv_mult_gpu(SEXP LinvSEXP, SEXP zSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type Linv(LinvSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(Linv_mult_gpu(Linv, z, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// exponential_isotropic
inline arma::mat exponential_isotropic(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_exponential_isotropic(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(exponential_isotropic(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_exponential_isotropic
inline arma::cube d_exponential_isotropic(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_d_exponential_isotropic(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_exponential_isotropic(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// matern_isotropic
inline arma::mat matern_isotropic(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_matern_isotropic(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_isotropic(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_matern_isotropic
inline arma::cube d_matern_isotropic(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_d_matern_isotropic(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_matern_isotropic(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// matern15_isotropic
inline arma::mat matern15_isotropic(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_matern15_isotropic(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(matern15_isotropic(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_matern15_isotropic
inline arma::cube d_matern15_isotropic(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_d_matern15_isotropic(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_matern15_isotropic(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// matern_scaledim
inline arma::mat matern_scaledim(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_matern_scaledim(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_scaledim(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_matern_scaledim
inline arma::cube d_matern_scaledim(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_d_matern_scaledim(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_matern_scaledim(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// exponential_scaledim
inline arma::mat exponential_scaledim(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_exponential_scaledim(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(exponential_scaledim(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_exponential_scaledim
inline arma::cube d_exponential_scaledim(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_d_exponential_scaledim(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_exponential_scaledim(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// matern_spacetime
inline arma::mat matern_spacetime(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_matern_spacetime(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_spacetime(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_matern_spacetime
inline arma::cube d_matern_spacetime(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_d_matern_spacetime(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_matern_spacetime(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// exponential_spacetime
inline arma::mat exponential_spacetime(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_exponential_spacetime(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(exponential_spacetime(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_exponential_spacetime
inline arma::cube d_exponential_spacetime(arma::vec covparms, arma::mat locs);
RcppExport SEXP _GpGpU_d_exponential_spacetime(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_exponential_spacetime(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// matern_spheretime
inline arma::mat matern_spheretime(arma::vec covparms, arma::mat lonlattime);
RcppExport SEXP _GpGpU_matern_spheretime(SEXP covparmsSEXP, SEXP lonlattimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type lonlattime(lonlattimeSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_spheretime(covparms, lonlattime));
    return rcpp_result_gen;
END_RCPP
}
// d_matern_spheretime
inline arma::cube d_matern_spheretime(arma::vec covparms, arma::mat lonlattime);
RcppExport SEXP _GpGpU_d_matern_spheretime(SEXP covparmsSEXP, SEXP lonlattimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type lonlattime(lonlattimeSEXP);
    rcpp_result_gen = Rcpp::wrap(d_matern_spheretime(covparms, lonlattime));
    return rcpp_result_gen;
END_RCPP
}
// exponential_spheretime
inline arma::mat exponential_spheretime(arma::vec covparms, arma::mat lonlattime);
RcppExport SEXP _GpGpU_exponential_spheretime(SEXP covparmsSEXP, SEXP lonlattimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type lonlattime(lonlattimeSEXP);
    rcpp_result_gen = Rcpp::wrap(exponential_spheretime(covparms, lonlattime));
    return rcpp_result_gen;
END_RCPP
}
// d_exponential_spheretime
inline arma::cube d_exponential_spheretime(arma::vec covparms, arma::mat lonlattime);
RcppExport SEXP _GpGpU_d_exponential_spheretime(SEXP covparmsSEXP, SEXP lonlattimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type lonlattime(lonlattimeSEXP);
    rcpp_result_gen = Rcpp::wrap(d_exponential_spheretime(covparms, lonlattime));
    return rcpp_result_gen;
END_RCPP
}
// Linv_mult
NumericVector Linv_mult(NumericMatrix Linv, NumericVector z, IntegerMatrix NNarray);
RcppExport SEXP _GpGpU_Linv_mult(SEXP LinvSEXP, SEXP zSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type Linv(LinvSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(Linv_mult(Linv, z, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// L_mult
NumericVector L_mult(NumericMatrix Linv, NumericVector z, IntegerMatrix NNarray);
RcppExport SEXP _GpGpU_L_mult(SEXP LinvSEXP, SEXP zSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type Linv(LinvSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(L_mult(Linv, z, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// Linv_t_mult
NumericVector Linv_t_mult(NumericMatrix Linv, NumericVector z, IntegerMatrix NNarray);
RcppExport SEXP _GpGpU_Linv_t_mult(SEXP LinvSEXP, SEXP zSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type Linv(LinvSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(Linv_t_mult(Linv, z, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// L_t_mult
NumericVector L_t_mult(NumericMatrix Linv, NumericVector z, IntegerMatrix NNarray);
RcppExport SEXP _GpGpU_L_t_mult(SEXP LinvSEXP, SEXP zSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type Linv(LinvSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(L_t_mult(Linv, z, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_Linv
NumericMatrix vecchia_Linv(arma::vec covparms, StringVector covfun_name, arma::mat locs, arma::mat NNarray, int start_ind);
RcppExport SEXP _GpGpU_vecchia_Linv(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP locsSEXP, SEXP NNarraySEXP, SEXP start_indSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type NNarray(NNarraySEXP);
    Rcpp::traits::input_parameter< int >::type start_ind(start_indSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_Linv(covparms, covfun_name, locs, NNarray, start_ind));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_profbeta_loglik_grad_info
List vecchia_profbeta_loglik_grad_info(NumericVector covparms, StringVector covfun_name, NumericVector y, NumericMatrix X, const NumericMatrix locs, NumericMatrix NNarray);
RcppExport SEXP _GpGpU_vecchia_profbeta_loglik_grad_info(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_profbeta_loglik_grad_info(covparms, covfun_name, y, X, locs, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_profbeta_loglik
List vecchia_profbeta_loglik(NumericVector covparms, StringVector covfun_name, NumericVector y, NumericMatrix X, const NumericMatrix locs, NumericMatrix NNarray);
RcppExport SEXP _GpGpU_vecchia_profbeta_loglik(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_profbeta_loglik(covparms, covfun_name, y, X, locs, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_meanzero_loglik
List vecchia_meanzero_loglik(NumericVector covparms, StringVector covfun_name, NumericVector y, const NumericMatrix locs, NumericMatrix NNarray);
RcppExport SEXP _GpGpU_vecchia_meanzero_loglik(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP locsSEXP, SEXP NNarraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type NNarray(NNarraySEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_meanzero_loglik(covparms, covfun_name, y, locs, NNarray));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_grouped_profbeta_loglik_grad_info
List vecchia_grouped_profbeta_loglik_grad_info(NumericVector covparms, StringVector covfun_name, NumericVector y, NumericMatrix X, const NumericMatrix locs, List NNlist);
RcppExport SEXP _GpGpU_vecchia_grouped_profbeta_loglik_grad_info(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNlistSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< List >::type NNlist(NNlistSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_grouped_profbeta_loglik_grad_info(covparms, covfun_name, y, X, locs, NNlist));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_grouped_profbeta_loglik
List vecchia_grouped_profbeta_loglik(NumericVector covparms, StringVector covfun_name, NumericVector y, NumericMatrix X, const NumericMatrix locs, List NNlist);
RcppExport SEXP _GpGpU_vecchia_grouped_profbeta_loglik(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNlistSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< List >::type NNlist(NNlistSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_grouped_profbeta_loglik(covparms, covfun_name, y, X, locs, NNlist));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_grouped_meanzero_loglik
List vecchia_grouped_meanzero_loglik(NumericVector covparms, StringVector covfun_name, NumericVector y, const NumericMatrix locs, List NNlist);
RcppExport SEXP _GpGpU_vecchia_grouped_meanzero_loglik(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP locsSEXP, SEXP NNlistSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< List >::type NNlist(NNlistSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_grouped_meanzero_loglik(covparms, covfun_name, y, locs, NNlist));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_GpGpU_nearest_neighbors_gpu", (DL_FUNC) &_GpGpU_nearest_neighbors_gpu, 2},
    {"_GpGpU_nearest_neighbors_sing_gpu", (DL_FUNC) &_GpGpU_nearest_neighbors_sing_gpu, 3},
    {"_GpGpU_vecchia_Linv_gpu", (DL_FUNC) &_GpGpU_vecchia_Linv_gpu, 4},
    {"_GpGpU_vecchia_Linv_gpu_isotropic_exponential_batched", (DL_FUNC) &_GpGpU_vecchia_Linv_gpu_isotropic_exponential_batched, 3},
    {"_GpGpU_vecchia_profbeta_loglik_grad_info_gpu", (DL_FUNC) &_GpGpU_vecchia_profbeta_loglik_grad_info_gpu, 6},
    {"_GpGpU_vecchia_profbeta_loglik_gpu", (DL_FUNC) &_GpGpU_vecchia_profbeta_loglik_gpu, 6},
    {"_GpGpU_fisher_scoring_gpu", (DL_FUNC) &_GpGpU_fisher_scoring_gpu, 9},
    {"_GpGpU_Linv_mult_gpu", (DL_FUNC) &_GpGpU_Linv_mult_gpu, 3},
    {"_GpGpU_exponential_isotropic", (DL_FUNC) &_GpGpU_exponential_isotropic, 2},
    {"_GpGpU_d_exponential_isotropic", (DL_FUNC) &_GpGpU_d_exponential_isotropic, 2},
    {"_GpGpU_matern_isotropic", (DL_FUNC) &_GpGpU_matern_isotropic, 2},
    {"_GpGpU_d_matern_isotropic", (DL_FUNC) &_GpGpU_d_matern_isotropic, 2},
    {"_GpGpU_matern15_isotropic", (DL_FUNC) &_GpGpU_matern15_isotropic, 2},
    {"_GpGpU_d_matern15_isotropic", (DL_FUNC) &_GpGpU_d_matern15_isotropic, 2},
    {"_GpGpU_matern_scaledim", (DL_FUNC) &_GpGpU_matern_scaledim, 2},
    {"_GpGpU_d_matern_scaledim", (DL_FUNC) &_GpGpU_d_matern_scaledim, 2},
    {"_GpGpU_exponential_scaledim", (DL_FUNC) &_GpGpU_exponential_scaledim, 2},
    {"_GpGpU_d_exponential_scaledim", (DL_FUNC) &_GpGpU_d_exponential_scaledim, 2},
    {"_GpGpU_matern_spacetime", (DL_FUNC) &_GpGpU_matern_spacetime, 2},
    {"_GpGpU_d_matern_spacetime", (DL_FUNC) &_GpGpU_d_matern_spacetime, 2},
    {"_GpGpU_exponential_spacetime", (DL_FUNC) &_GpGpU_exponential_spacetime, 2},
    {"_GpGpU_d_exponential_spacetime", (DL_FUNC) &_GpGpU_d_exponential_spacetime, 2},
    {"_GpGpU_matern_spheretime", (DL_FUNC) &_GpGpU_matern_spheretime, 2},
    {"_GpGpU_d_matern_spheretime", (DL_FUNC) &_GpGpU_d_matern_spheretime, 2},
    {"_GpGpU_exponential_spheretime", (DL_FUNC) &_GpGpU_exponential_spheretime, 2},
    {"_GpGpU_d_exponential_spheretime", (DL_FUNC) &_GpGpU_d_exponential_spheretime, 2},
    {"_GpGpU_Linv_mult", (DL_FUNC) &_GpGpU_Linv_mult, 3},
    {"_GpGpU_L_mult", (DL_FUNC) &_GpGpU_L_mult, 3},
    {"_GpGpU_Linv_t_mult", (DL_FUNC) &_GpGpU_Linv_t_mult, 3},
    {"_GpGpU_L_t_mult", (DL_FUNC) &_GpGpU_L_t_mult, 3},
    {"_GpGpU_vecchia_Linv", (DL_FUNC) &_GpGpU_vecchia_Linv, 5},
    {"_GpGpU_vecchia_profbeta_loglik_grad_info", (DL_FUNC) &_GpGpU_vecchia_profbeta_loglik_grad_info, 6},
    {"_GpGpU_vecchia_profbeta_loglik", (DL_FUNC) &_GpGpU_vecchia_profbeta_loglik, 6},
    {"_GpGpU_vecchia_meanzero_loglik", (DL_FUNC) &_GpGpU_vecchia_meanzero_loglik, 5},
    {"_GpGpU_vecchia_grouped_profbeta_loglik_grad_info", (DL_FUNC) &_GpGpU_vecchia_grouped_profbeta_loglik_grad_info, 6},
    {"_GpGpU_vecchia_grouped_profbeta_loglik", (DL_FUNC) &_GpGpU_vecchia_grouped_profbeta_loglik, 6},
    {"_GpGpU_vecchia_grouped_meanzero_loglik", (DL_FUNC) &_GpGpU_vecchia_grouped_meanzero_loglik, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_GpGpU(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
