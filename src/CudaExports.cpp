// #include <Rcpp.h>
#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
#include <chrono>
using namespace Rcpp;
using namespace arma;


#include "covmatrix_funs.h"

extern "C"
void call_compute_pieces_gpu(
    double* covparms,
    const short covfun_name,
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

extern "C"
void call_compute_pieces_gpu_batched(
    double* covparms,
    const short covfun_name,
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


arma::vec forward_solve2( arma::mat cholmat, arma::vec b ){

    int n = cholmat.n_rows;
    arma::vec x(n);
    x(0) = b(0)/cholmat(0,0);

    for(int i=1; i<n; i++){
        double dd = 0.0;
        for(int j=0; j<i; j++){
            dd += cholmat(i,j)*x(j);
        }
        x(i) = (b(i)-dd)/cholmat(i,i);
    }    
    return x;

} 
arma::mat forward_solve_mat2( arma::mat cholmat, arma::mat b ){

    int n = cholmat.n_rows;
    int p = b.n_cols;
    arma::mat x(n,p);
    for(int k=0; k<p; k++){ x(0,k) = b(0,k)/cholmat(0,0); }

    for(int i=1; i<n; i++){
	for(int k=0; k<p; k++){
            double dd = 0.0;
            for(int j=0; j<i; j++){
                dd += cholmat(i,j)*x(j,k);
            }
            x(i,k) = (b(i,k)-dd)/cholmat(i,i);
       	}
    }    
    return x;
} 
arma::vec backward_solve2( arma::mat lower, arma::vec b ){

    int n = lower.n_rows;
    arma::vec x(n);
    x(n-1) = b(n-1)/lower(n-1,n-1);

    for(int i=n-2; i>=0; i--){
        double dd = 0.0;
        for(int j=n-1; j>i; j--){
            dd += lower(j,i)*x(j);
        }
        x(i) = (b(i)-dd)/lower(i,i);
    }    
    return x;
} 
arma::mat backward_solve_mat2( arma::mat cholmat, arma::mat b ){

    int n = cholmat.n_rows;
    int p = b.n_cols;
    arma::mat x(n,p);
    for(int k=0; k<p; k++){ x(n-1,k) = b(n-1,k)/cholmat(n-1,n-1); }

    for(int i=n-2; i>=0; i--){
	for(int k=0; k<p; k++){
            double dd = 0.0;
            for(int j=n-1; j>i; j--){
                dd += cholmat(j,i)*x(j,k);
            }
            x(i,k) = (b(i,k)-dd)/cholmat(i,i);
       	}
    }    
    return x;
} 


arma::mat mychol2( arma::mat A ){

    arma::uword n = A.n_rows;
    arma::mat L(n,n);
    bool pd = true;
    
    // upper-left entry
    if( A(0,0) < 0 ){
	pd = false;
	L(0,0) = 1.0;
    } else {
        L(0,0) = std::sqrt(A(0,0));
    }
    if( n > 1 ){
	// second row
	L(1,0) = A(1,0)/L(0,0);
	double f = A(1,1) - L(1,0)*L(1,0);
	if( f < 0 ){
	    pd = false;
	    L(1,1) = 1.0;
	} else {
	    L(1,1) = std::sqrt( f );
	}
	// rest of the rows
	if( n > 2 ){
            for(uword i=2; i<n; i++){
    	        // leftmost entry in row i
    	        L(i,0) = A(i,0)/L(0,0);
    	        // middle entries in row i 
    	        for(uword j=1; j<i; j++){
    	            double d = A(i,j);
    	            for(uword k=0; k<j; k++){
    	        	d -= L(i,k)*L(j,k);
    	            }
    	            L(i,j) = d/L(j,j);
    	        }
		// diagonal entry in row i
    	        double e = A(i,i);
    	        for(uword k=0; k<i; k++){
    	            e -= L(i,k)*L(i,k);
    	        }
		if( e < 0 ){
		    pd = false;
		    L(i,i) = 1.0;
		} else {
    	            L(i,i) = std::sqrt(e);
		}
	    }
	}
    }
    return L;	
}

void compute_pieces2(
    arma::vec covparms, 
    StringVector covfun_name,
    arma::mat locs, 
    arma::mat NNarray,
    arma::vec y, 
    arma::mat X,
    mat* XSX,
    vec* ySX,
    double* ySy,
    double* logdet,
    cube* dXSX,
    mat* dySX,
    vec* dySy,
    vec* dlogdet,
    mat* ainfo,
    int profbeta,
    int grad_info
){
    // printf("compute_pieces2\n");
    // data dimensions
    int n = y.n_elem;
    int m = NNarray.n_cols;
    int p = X.n_cols;
    int nparms = covparms.n_elem;
    int dim = locs.n_cols;
    // convert StringVector to std::string to use .compare() below
    std::string covfun_name_string;
    covfun_name_string = covfun_name[0];
    
    // assign covariance fun and derivative based on covfun_name_string

    /* p_covfun is an array of length 1. Its entry is a pointer to a function which takes
     in arma::vec and arma::mat and returns mat. p_d_covfun is analogous. This was a workaround for the solaris bug*/

    mat (*p_covfun[1])(arma::vec, arma::mat);
    cube (*p_d_covfun[1])(arma::vec, arma::mat);
    get_covfun(covfun_name_string, p_covfun, p_d_covfun);
    

#pragma omp parallel 
{   
    arma::mat l_XSX = arma::mat(p, p, fill::zeros);
    arma::vec l_ySX = arma::vec(p, fill::zeros);
    double l_ySy = 0.0;
    double l_logdet = 0.0;
    arma::cube l_dXSX = arma::cube(p,p, nparms, fill::zeros);
    arma::mat l_dySX = arma::mat(p, nparms, fill::zeros);
    arma::vec l_dySy = arma::vec(nparms, fill::zeros);
    arma::vec l_dlogdet = arma::vec(nparms, fill::zeros);
    arma::mat l_ainfo = arma::mat(nparms, nparms, fill::zeros);

    #pragma omp for	    
    for(int i=0; i<m; i++){
        int bsize = std::min(i+1,m);

	//std::vector<std::chrono::steady_clock::time_point> tt;

	//tt.push_back( std::chrono::steady_clock::now() );

        // first, fill in ysub, locsub, and X0 in reverse order
        arma::mat locsub(bsize, dim);
        arma::vec ysub(bsize);
        arma::mat X0( bsize, p );
        for(int j=bsize-1; j>=0; j--){
            ysub(bsize-1-j) = y( NNarray(i,j)-1 );
            for(int k=0;k<dim;k++){ locsub(bsize-1-j,k) = locs( NNarray(i,j)-1, k ); }
            if(profbeta){ 
                for(int k=0;k<p;k++){ X0(bsize-1-j,k) = X( NNarray(i,j)-1, k ); } 
            }
        }
        // compute covariance matrix and derivatives and take cholesky
        arma::mat covmat = p_covfun[0]( covparms, locsub );	
        // arma::mat covmat = exponential_isotropic2(covparms, locsub);
        arma::cube dcovmat;
        if(grad_info){ 
            // dcovmat = d_exponential_isotropic2( covparms, locsub ); 
            dcovmat = p_d_covfun[0]( covparms, locsub ); 
        }
        arma::mat cholmat = eye( size(covmat) );
        chol( cholmat, covmat, "lower" );
        // i1 is conditioning set, i2 is response        
        //arma::span i1 = span(0,bsize-2);
        arma::span i2 = span(bsize-1,bsize-1);
        // get last row of cholmat
        arma::vec onevec = zeros(bsize);
        onevec(bsize-1) = 1.0;
        arma::vec choli2;
        if(grad_info){
            //choli2 = solve( trimatu(cholmat.t()), onevec );
            choli2 = backward_solve2( cholmat, onevec );
        }
        bool cond = bsize > 1;
        //double fac = 1.0;
        
        // do solves with X and y
        arma::mat LiX0;
        if(profbeta){
            //LiX0 = solve( trimatl(cholmat), X0 );
            LiX0 = forward_solve_mat2( cholmat, X0 );
        }
        //arma::vec Liy0 = solve( trimatl(cholmat), ysub );
        arma::vec Liy0 = forward_solve2( cholmat, ysub );
        
        // loglik objects
        l_logdet += 2.0*std::log( as_scalar(cholmat(i2,i2)) ); 
        l_ySy +=    pow( as_scalar(Liy0(i2)), 2 );
        if(profbeta){
            l_XSX +=   LiX0.rows(i2).t() * LiX0.rows(i2);
            l_ySX += ( Liy0(i2) * LiX0.rows(i2) ).t();
        }
        if( grad_info ){
        // gradient objects
        // LidSLi3 is last column of Li * (dS_j) * Lit for 1 parameter i
        // LidSLi2 stores these columns in a matrix for all parameters
        arma::mat LidSLi2(bsize,nparms);
        
        if(cond){ // if we condition on anything
            
            for(int j=0; j<nparms; j++){
                // compute last column of Li * (dS_j) * Lit
                //arma::vec LidSLi3 = solve( trimatl(cholmat), dcovmat.slice(j) * choli2 );
                arma::vec LidSLi3 = forward_solve2( cholmat, dcovmat.slice(j) * choli2 );
                // store LiX0.t() * LidSLi3 and Liy0.t() * LidSLi3
                arma::vec v1 = LiX0.t() * LidSLi3;
                double s1 = as_scalar( Liy0.t() * LidSLi3 ); 
                // update all quantities
                // bottom-right corner gets double counted, so need to subtract it off
                (l_dXSX).slice(j) += v1 * LiX0.rows(i2) + ( v1 * LiX0.rows(i2) ).t() - 
                    as_scalar(LidSLi3(i2)) * ( LiX0.rows(i2).t() * LiX0.rows(i2) );
                (l_dySy)(j) += as_scalar( 2.0 * s1 * Liy0(i2)  - 
                    LidSLi3(i2) * Liy0(i2) * Liy0(i2) );
                (l_dySX).col(j) += (  s1 * LiX0.rows(i2) + ( v1 * Liy0(i2) ).t() -  
                    as_scalar( LidSLi3(i2) ) * LiX0.rows(i2) * as_scalar( Liy0(i2))).t();
                (l_dlogdet)(j) += as_scalar( LidSLi3(i2) );
                // store last column of Li * (dS_j) * Lit
                LidSLi2.col(j) = LidSLi3;
            }
            // fisher information object
            // bottom right corner gets double counted, so subtract it off
            for(int i=0; i<nparms; i++){ for(int j=0; j<i+1; j++){
                (l_ainfo)(i,j) += 
                    1.0*accu( LidSLi2.col(i) % LidSLi2.col(j) ) - 
                    0.5*accu( LidSLi2.rows(i2).col(j) %
                              LidSLi2.rows(i2).col(i) );
            }}
            
        } else { // similar calculations, but for when there is no conditioning set
            for(int j=0; j<nparms; j++){
                //arma::mat LidSLi = solve( trimatl(cholmat), dcovmat.slice(j) );
                arma::mat LidSLi = forward_solve_mat2( cholmat, dcovmat.slice(j) );
                //LidSLi = solve( trimatl(cholmat), LidSLi.t() );
                LidSLi = forward_solve_mat2( cholmat, LidSLi.t() );
                (l_dXSX).slice(j) += LiX0.t() *  LidSLi * LiX0; 
                (l_dySy)(j) += as_scalar( Liy0.t() * LidSLi * Liy0 );
                (l_dySX).col(j) += ( ( Liy0.t() * LidSLi ) * LiX0 ).t();
                (l_dlogdet)(j) += trace( LidSLi );
                LidSLi2.col(j) = LidSLi;
            }
            
            // fisher information object
            for(int i=0; i<nparms; i++){ for(int j=0; j<i+1; j++){
                (l_ainfo)(i,j) += 0.5*accu( LidSLi2.col(i) % LidSLi2.col(j) ); 
            }}

        }
        
    }
}
#pragma omp critical
{
    *XSX += l_XSX;
    *ySX += l_ySX;
    *ySy += l_ySy;
    *logdet += l_logdet;
    *dXSX += l_dXSX;
    *dySX += l_dySX;
    *dySy += l_dySy;
    *dlogdet += l_dlogdet;
    *ainfo += l_ainfo;
}
}
    // printf("compute_pieces2...done\n");
}    

void likelihood(
    NumericVector covparms,
    StringVector covfun_name,
    NumericMatrix locs,
    NumericMatrix NNarray,
    NumericVector y,
    NumericMatrix X,
    NumericVector* ll, 
    NumericVector* betahat,
    NumericVector* grad,
    NumericMatrix* info,
    NumericMatrix* betainfo,
    bool profbeta,
    bool grad_info
) {
    
    short covfun_name_index = 0;
    std::string covfun_name_string;
    covfun_name_string = covfun_name[0];
    if (covfun_name_string.compare("exponential_isotropic") == 0) {
        covfun_name_index = 0;
    } else if (covfun_name_string.compare("exponential_scaledim") == 0) {
        covfun_name_index = 1;
    } else if (covfun_name_string.compare("exponential_spacetime") == 0) {
        covfun_name_index = 2;
    } else if (covfun_name_string.compare("exponential_spheretime") == 0) {
        covfun_name_index = 3;
    }

    int n = y.length();
    int m = NNarray.ncol();
    int p = X.ncol();
    int nparms = covparms.length();
    int dim = locs.ncol();
    // int dim = locs.n_cols;

    double* covparmsl = (double*)malloc(sizeof(double) * nparms);

    for (int i = 0; i < nparms; i++) {
        covparmsl[i] = covparms[i];
    }
    
    double* locsl = (double*)malloc(sizeof(double) * n * dim);
    
    // locs = locs.t();
    // double* locsl = locs.memptr();
    double* NNarrayl = (double*) calloc(n * m, sizeof(double));//malloc(sizeof(double) * n * m);
    double* yl = (double*)malloc(sizeof(double) * n);
    double* Xl = (double*)malloc(sizeof(double) * n * p);
    
    for (int i = 0; i < n; i++) {
        yl[i] = y[i];
        for (int j = 0; j < std::max(m, std::max(p, dim)); j++) {
            if (j < dim) {
                locsl[i * dim + j] = locs(i, j);
            }
            if (j < p) {
                Xl[i * p + j] = X(i, j);
            }
            if (j <= i && j < m){
                NNarrayl[i * m + j] = NNarray(i, j);
            }
        }
    }

    double* XSXf = (double*)calloc(p * p, sizeof(double));
    double* ySXf = (double*)calloc(p * p, sizeof(double));
    double ySy = 0;
    double logdet = 0;

    double* dXSXf = (double*)calloc(p * p * nparms, sizeof(double));
    double* dySXf = (double*)calloc(p * nparms, sizeof(double));
    double* dySyf = (double*)calloc(nparms, sizeof(double));
    double* dlogdetf = (double*)calloc(nparms, sizeof(double));
    double* ainfof = (double*)calloc(nparms * nparms, sizeof(double));
    call_compute_pieces_gpu(covparmsl, covfun_name_index, locsl, NNarrayl, yl, Xl,
        XSXf, ySXf, &ySy, &logdet,
        dXSXf, dySXf, dySyf, dlogdetf, ainfof, profbeta, grad_info, n, m, p, nparms, dim);
    // call_compute_pieces_gpu_batched(covparmsl, covfun_name_index, locsl, NNarrayl, yl, Xl,
    //     XSXf, ySXf, &ySy, &logdet,
    //     dXSXf, dySXf, dySyf, dlogdetf, ainfof, profbeta, grad_info, n, m, p, nparms, dim);

    arma::mat XSX = arma::mat(&XSXf[0], p, p);
    arma::vec ySX = arma::vec(&ySXf[0], p);
    arma::mat dySX = arma::mat(&dySXf[0], nparms, p);
    // arma::cube dXSX = arma::cube(&dXSXf[0], p, p, nparms);
    arma::cube dXSX(p, p, nparms, fill::zeros);
    for(int j=0; j<nparms; j++){
        for (int i = 0; i < p; i++){
            for (int k = 0; k < p; k++){
                dXSX.slice(j)(i, k) = dXSXf[k * p * nparms + i * nparms + j];
            }
        }
    }
    
    XSX = XSX.t();
    // ySX = ySX.t();
    dySX = dySX.t();
    arma::vec dlogdet = arma::vec(&dlogdetf[0], nparms);
    arma::vec dySy = arma::vec(&dySyf[0], nparms);
    arma::mat ainfo = arma::mat(&ainfof[0], nparms, nparms);
    ainfo = ainfo.t();

    //first m

    arma::mat XSXl = arma::mat(p, p, fill::zeros);
    arma::vec ySXl = arma::vec(p, fill::zeros);
    arma::cube dXSXl = arma::cube(p,p,nparms,fill::zeros);
    arma::mat dySXl = arma::mat(p, nparms, fill::zeros);
    arma::vec dySyl = arma::vec(nparms, fill::zeros);
    arma::vec dlogdetl = arma::vec(nparms, fill::zeros);
    // fisher information
    arma::mat ainfol = arma::mat(nparms, nparms, fill::zeros);

    double ySyl = 0;
    double logdetl = 0;

    arma::vec covparms_c = arma::vec(covparms.begin(),covparms.length());
    arma::mat locs_c = arma::mat(locs.begin(), locs.nrow(), locs.ncol());//locsm; 
    arma::mat NNarray_c = arma::mat(NNarray.begin(), NNarray.nrow(), NNarray.ncol());
    arma::vec y_c = arma::vec(y.begin(),y.length());
    arma::mat X_c = arma::mat(X.begin(),X.nrow(),X.ncol());

    compute_pieces2(covparms_c, covfun_name, locs_c, NNarray_c, y_c, X_c, 
        &XSXl, &ySXl, &ySyl, &logdetl, &dXSXl, &dySXl, &dySyl,
        &dlogdetl, &ainfol, profbeta, grad_info);
    XSX += XSXl;
    ySX += ySXl;
    ySy += ySyl;
    logdet += logdetl;
    dXSX += dXSXl;
    dySX += dySXl;
    dySy += dySyl;
    dlogdet += dlogdetl;
    ainfo += ainfol;
    
    // synthesize everything and update loglik, grad, beta, betainfo, info
    // betahat and dbeta
    arma::vec abeta = arma::vec( p, fill::zeros );
    if(profbeta){ abeta = solve( XSX, ySX ); }
    
    for(int j=0; j<p; j++){ (*betahat)(j) = abeta(j); };
    arma::mat dbeta = arma::mat(p,nparms, fill::zeros);
    
    if( profbeta && grad_info){
        for(int j=0; j<nparms; j++){
            dbeta.col(j) = solve( XSX, dySX.col(j) - dXSX.slice(j) * abeta );
        }
    }
    // get sigmahatsq
    double sig2 = ( ySy - 2.0*as_scalar( ySX.t() * abeta ) + 
        as_scalar( abeta.t() * XSX * abeta ) )/n;
    // loglikelihood
    (*ll)(0) = -0.5*( n*std::log(2.0*M_PI) + logdet + n*sig2 ); 
    if(profbeta){
    // betainfo
    for(int i=0; i<p; i++){ for(int j=0; j<i+1; j++){
        (*betainfo)(i,j) = XSX(i,j);
        (*betainfo)(j,i) = XSX(j,i);
    }}
    }

    if(grad_info){
    // gradient
    for(int j=0; j<nparms; j++){
        (*grad)(j) = 0.0;
        (*grad)(j) -= 0.5*dlogdet(j);
        (*grad)(j) += 0.5*dySy(j);
        (*grad)(j) -= 1.0*as_scalar( abeta.t() * dySX.col(j) );
        (*grad)(j) += 1.0*as_scalar( ySX.t() * dbeta.col(j) );
        (*grad)(j) += 0.5*as_scalar( abeta.t() * dXSX.slice(j) * abeta );
        (*grad)(j) -= 1.0*as_scalar( abeta.t() * XSX * dbeta.col(j) );
    }

    // fisher information
    for(int i=0; i<nparms; i++){ for(int j=0; j<i+1; j++){
        (*info)(i,j) = ainfo(i,j);
        (*info)(j,i) = (*info)(i,j);
    }}
    }
    
}

// [[Rcpp::export]]
List vecchia_profbeta_loglik_grad_info_gpu( 
    NumericVector covparms, 
    StringVector covfun_name,
    NumericVector y,
    NumericMatrix X,
    NumericMatrix locs,
    NumericMatrix NNarray ){
    
    NumericVector ll(1);
    NumericVector grad( covparms.length() );
    NumericVector betahat( X.ncol() );
    NumericMatrix info( covparms.length(), covparms.length() );
    NumericMatrix betainfo( X.ncol(), X.ncol() );

    // this function calls arma_onepass_compute_pieces
    // then synthesizes the result into loglik, beta, grad, info, betainfo
    likelihood(covparms, covfun_name, locs, NNarray, y, X,
        &ll, &betahat, &grad, &info, &betainfo, true, true 
    );
    
    List ret = List::create( Named("loglik") = ll, Named("betahat") = betahat,
        Named("grad") = grad, Named("info") = info, Named("betainfo") = betainfo );
    return ret;
        
}

// [[Rcpp::export]]
List vecchia_profbeta_loglik_gpu( 
    NumericVector covparms, 
    StringVector covfun_name,
    NumericVector y,
    NumericMatrix X,
    NumericMatrix locs,
    NumericMatrix NNarray ){
    
    NumericVector ll(1);
    NumericVector grad( covparms.length() );
    NumericVector betahat( X.ncol() );
    NumericMatrix info( covparms.length(), covparms.length() );
    NumericMatrix betainfo( X.ncol(), X.ncol() );

    // this function calls arma_onepass_compute_pieces
    // then synthesizes the result into loglik, beta, grad, info, betainfo
    likelihood(covparms, covfun_name, locs, NNarray, y, X,
        &ll, &betahat, &grad, &info, &betainfo, true, false 
    );
    
    List ret = List::create( Named("loglik") = ll, Named("betahat") = betahat,
        Named("betainfo") = betainfo );
    return ret;
        
}

// [[Rcpp::export]]
List vecchia_meanzero_loglik_gpu( 
    NumericVector covparms, 
    StringVector covfun_name,
    NumericVector y,
    NumericMatrix X1,
    NumericMatrix locs,
    NumericMatrix NNarray ){
    
    NumericMatrix X(1,1);
    NumericVector ll(1);
    NumericVector grad( covparms.length() );
    NumericVector betahat( X.ncol() );
    NumericMatrix info( covparms.length(), covparms.length() );
    NumericMatrix betainfo( X.ncol(), X.ncol() );

    // this function calls arma_onepass_compute_pieces
    // then synthesizes the result into loglik, beta, grad, info, betainfo
    likelihood(covparms, covfun_name, locs, NNarray, y, X,
        &ll, &betahat, &grad, &info, &betainfo, false, false 
    );
    
    List ret = List::create( Named("loglik") = ll);
    return ret;
        
}
