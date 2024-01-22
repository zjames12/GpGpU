// #include <Rcpp.h>
#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


#include "covmatrix_funs.h"


extern "C"
double* vecchia_Linv_gpu_outer(
    double* covparms,
    double* locs,
    double* NNarray,
    int n,
    int m,
    int dim);

extern "C"
double* vecchia_Linv_gpu_batched(
    double* covparms,
    double* locs,
    double* NNarray,
    int n,
    int m,
    int dim);

extern "C"
double** load_data(
    const double* locs,
    const double* NNarray,
    const double* y,
    const double* X,
    int n, int m, int dim, int p, int nparms);

extern "C"
void free_data(double** data_stores);

extern "C"
int* nearest_neighbors(float* locs, int m, int n, int dim);

extern "C"
int* nearest_neighbors_sing(float* locs, int m, int n, int dim, int nq);

// [[Rcpp::export]]
NumericMatrix nearest_neighbors_gpu(arma::mat locs, int m){
    int n = locs.n_rows;
    int dim = locs.n_cols;

    // locs = locs.t();
    // double* locsl = locs.memptr();
    float* locsl = (float*) malloc(sizeof(float) * n * dim);
    for (int i = 0; i < n; i++) {
		for (int j = 0; j < dim; j++){
			locsl[i * dim + j] = locs(i, j);
		}
	}

    int* NNarrayl = nearest_neighbors(locsl, m, n, dim);
    // arma::mat NNarray = arma::mat(&NNarrayl[0], m + 1, n, false);
    NumericMatrix NNarray(n, m + 1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m + 1; j++){
			NNarray(i, j) = NNarrayl[i * (m + 1) + j];
		}
	}
	return NNarray;
}

// [[Rcpp::export]]
NumericMatrix nearest_neighbors_sing_gpu(arma::mat locs, int m, int nq){
    int n = locs.n_rows;
    int dim = locs.n_cols;

    // locs = locs.t();
    // double* locsl = locs.memptr();
    float* locsl = (float*) malloc(sizeof(float) * n * dim);
    for (int i = 0; i < n; i++) {
		for (int j = 0; j < dim; j++){
			locsl[i * dim + j] = locs(i, j);
		}
	}

    int* NNarrayl = nearest_neighbors_sing(locsl, m, n, dim, nq);
    // arma::mat NNarray = arma::mat(&NNarrayl[0], m + 1, n, false);
    NumericMatrix NNarray(n, m + 1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m + 1; j++){
			NNarray(i, j) = NNarrayl[i * (m + 1) + j];
		}
	}
	return NNarray;
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

// [[Rcpp::export]]
arma::mat vecchia_Linv_gpu_isotropic_exponential_batched(
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

    double* Linvl = vecchia_Linv_gpu_batched(covparmsl, locsl, NNarrayl, n, m, dim);

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
double* call_Linv_mult_gpu(double* Linv, double* z, double* NNarray, int n, int m);

arma::mat exponential_isotropic2(arma::vec covparms, arma::mat locs ){

    int dim = locs.n_cols;
    int n = locs.n_rows;
    double nugget = covparms( 0 )*covparms( 2 );
    
    // create scaled locations
    mat locs_scaled(n,dim);
    for(int j=0; j<dim; j++){ 
        for(int i=0; i<n; i++){
            locs_scaled(i,j) = locs(i,j)/covparms(1);
        }
    }
    // calculate covariances
    arma::mat covmat(n,n);
    for(int i1=0; i1<n; i1++){ for(int i2=0; i2<=i1; i2++){
        // calculate distance
        double d = 0.0;
        for(int j=0; j<dim; j++){
            d += pow( locs_scaled(i1,j) - locs_scaled(i2,j), 2.0 );
        }
        d = std::sqrt( d );

        // calculate covariance            
        covmat(i2,i1) = covparms(0)*std::exp( -d );
        covmat(i1,i2) = covmat(i2,i1);

    }}

    // add nugget
    for(int i1=0; i1<n; i1++){
	covmat(i1,i1) += nugget;
    }

    return covmat;
}
arma::cube d_exponential_isotropic2(arma::vec covparms, arma::mat locs ){

    int dim = locs.n_cols;
    int n = locs.n_rows;
    //double nugget = covparms( 0 )*covparms( 2 );
    // create scaled locations
    mat locs_scaled(n,dim);
    for(int j=0; j<dim; j++){ 
        for(int i=0; i<n; i++){
            locs_scaled(i,j) = locs(i,j)/covparms(1);
        }
    }
    // calculate derivatives
    arma::cube dcovmat = arma::cube(n,n,covparms.n_elem, fill::zeros);
    for(int i1=0; i1<n; i1++){ for(int i2=0; i2<=i1; i2++){
        double d = 0.0;
        for(int j=0; j<dim; j++){
            d += pow( locs_scaled(i1,j) - locs_scaled(i2,j), 2.0 );
        }
        d = std::sqrt( d );
        
        dcovmat(i1,i2,0) += std::exp(-d);
        dcovmat(i1,i2,1) += covparms(0)*std::exp(-d)*d/covparms(1);
        if( i1 == i2 ){ // update diagonal entry
            dcovmat(i1,i2,0) += covparms(2);
            dcovmat(i1,i2,2) += covparms(0); 
        } else { // fill in opposite entry
            for(int j=0; j<covparms.n_elem; j++){
                dcovmat(i2,i1,j) = dcovmat(i1,i2,j);
            }
        }
    }}

    return dcovmat;
}
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
	// if(i % 100 == 0 ){
	//     for(int k=1; k<tt.size(); k++ ){
	//         cout << std::chrono::duration_cast<std::chrono::microseconds>(tt[k]-tt[k-1]).count();
	//         cout << "  ";
	//     }
	//     cout << endl;
	// }


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
    }

    int n = y.length();
    int m = NNarray.ncol();
    int p = X.ncol();
    int nparms = covparms.length();
    int dim = locs.ncol();

    double* covparmsl = (double*)malloc(sizeof(double) * nparms);
    // covparmsl[0] = covparms[0];
    // covparmsl[1] = covparms[1];
    // covparmsl[2] = covparms[2];
    for (int i = 0; i < nparms; i++) {
        covparmsl[i] = covparms[i];
    }
    
    double* locsl = (double*)malloc(sizeof(double) * n * dim);
    double* NNarrayl = (double*) calloc(n * m, sizeof(double));//malloc(sizeof(double) * n * m);
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
            if (j <= i){
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

    // printf("XSXf %f\n", XSXf[0]);
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

    // printf("XSX\n");
    // XSX.print();
    // printf("ySX\n");
    // ySX.print();
    // printf("ySy: %f\n", ySy);
    // // ySyl.print();
    // printf("logdet: %f\n", logdet);
    // // logdetl.print();
    // printf("dXSX\n");
    // dXSX.print();
    // printf("dySX\n");
    // dySX.print();
    // printf("dySy\n");
    // dySy.print();
    // printf("dlogdet\n");
    // dlogdet.print();
    // printf("ainfo\n");
    // ainfo.print();

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
    arma::mat locs_c = arma::mat(locs.begin(), locs.nrow(), locs.ncol());
    arma::mat NNarray_c = arma::mat(NNarray.begin(), NNarray.nrow(), NNarray.ncol());
    arma::vec y_c = arma::vec(y.begin(),y.length());
    arma::mat X_c = arma::mat(X.begin(),X.nrow(),X.ncol());

    // printf("covparms_c\n");
    // covparms_c.print();
    // printf("locs_c\n");
    // locs_c.print();
    // printf("NNarray_c: %f\n", ySyl);
    // NNarray_c.print();
    // printf("y_c: %f\n", logdetl);
    // y_c.print();
    // printf("X_c\n");
    // X_c.print();

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


    
    // NumericVector ll(1);
    // NumericVector grad( covparms.length() );
    // NumericVector betahat( X.ncol() );
    // NumericMatrix info( covparms.length(), covparms.length() );
    // NumericMatrix betainfo( X.ncol(), X.ncol() );
    
    // synthesize everything and update loglik, grad, beta, betainfo, info
    
    // betahat and dbeta
    arma::vec abeta = arma::vec( p, fill::zeros );
    // XSX.print();
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
    // printf("dlogdet\n");
    // dlogdet.print();
    // printf("dySy\n");
    // dySy.print();
    // printf("abeta\n");
    // abeta.print();
    // printf("dySX\n");
    // dySX.print();
    // printf("ySX\n");
    // ySX.print();
    // printf("dbeta\n");
    // dbeta.print();
    // printf("dXSX\n");
    // dXSX.print();

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
    const NumericMatrix locs,
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
    const NumericMatrix locs,
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

extern "C"
void call_compute_pieces_fisher_gpu(
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
    int dim,
    double** data_store
);

void likfun(arma::vec* grad, arma::mat* info, arma::mat* betainfo, arma::vec* ll, arma::vec* betahat,
    double* locsl, double* NNarrayl, double* yl, double* Xl,
    arma::vec dlogdet, arma::vec dySy, arma::mat ainfo, arma::mat XSXl, arma::vec ySXl, arma::cube dXSXl,
    arma::vec dySyl, arma::vec dlogdetl, arma::mat ainfol, arma::mat dySXl,
    arma::vec covparms_c, arma::mat locs_c, arma::mat NNarray_c, arma::vec y_c, arma::mat X_c,
    double* XSXf, double* ySXf, double* dXSXf, double* dySXf, 
    double* dySyf, double* dlogdetf, double* ainfof, bool profbeta, bool grad_info, 
    int n, int m, int p, int nparms, int dim, double** loaded_data){

    // *ll = arma::vec(1, fill::zeros);
    // *grad = arma::vec(nparms, fill::zeros);
    // *betahat = arma::vec(p, fill::zeros);
    // *info = arma::mat(nparms, nparms, fill::zeros);
    // *betainfo = arma::mat(p, p, fill::zeros);

    // XSXf = (double*)calloc(p * p, sizeof(double));
    // ySXf = (double*)calloc(p * p, sizeof(double));


    // dXSXf = (double*)calloc(p * p * nparms, sizeof(double));
    // dySXf = (double*)calloc(p * nparms, sizeof(double));
    // dySyf = (double*)calloc(nparms, sizeof(double));
    // dlogdetf = (double*)calloc(nparms, sizeof(double));
    // ainfof = (double*)calloc(nparms * nparms, sizeof(double));

    
    // dlogdet = arma::vec(&dlogdetf[0], nparms);
    // dySy = arma::vec(&dySyf[0], nparms);
    // ainfo = arma::mat(&ainfof[0], nparms, nparms);
    // ainfo = ainfo.t();

    // // first m

    // XSXl = arma::mat(p, p, fill::zeros);
    // ySXl = arma::vec(p, fill::zeros);
    // dXSXl = arma::cube(p,p,nparms,fill::zeros);
    // dySXl = arma::mat(p, nparms, fill::zeros);
    // dySyl = arma::vec(nparms, fill::zeros);
    // dlogdetl = arma::vec(nparms, fill::zeros);
    // // fisher information
    // ainfol = arma::mat(nparms, nparms, fill::zeros);
    // // // printf("likfun\n");

    double ySy = 0;
    double logdet = 0;

    double covparmsl[3] = { 0, 0, 0 };
    covparmsl[0] = covparms_c[0];
    covparmsl[1] = covparms_c[1];
    covparmsl[2] = covparms_c[2];

    

    call_compute_pieces_fisher_gpu(covparmsl, locsl, NNarrayl, yl, Xl,
            XSXf, ySXf, &ySy, &logdet,
            dXSXf, dySXf, dySyf, dlogdetf, ainfof, profbeta, grad_info, n, m, p, nparms, dim, loaded_data);
    
    arma::mat XSX = arma::mat(&XSXf[0], p, p);
    arma::vec ySX = arma::vec(&ySXf[0], p);
    arma::mat dySX = arma::mat(&dySXf[0], nparms, p);
    
    // arma::cube dXSX = arma::cube(&dXSXf[0], p, p, nparms);
    arma::cube dXSX = arma::cube(p, p, nparms, fill::zeros);
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

    dlogdet = arma::vec(&dlogdetf[0], nparms);
    dySy = arma::vec(&dySyf[0], nparms);
    ainfo = arma::mat(&ainfof[0], nparms, nparms);
    ainfo = ainfo.t();
    

    double ySyl = 0;
    double logdetl = 0;



    // compute_pieces2(covparms_c, "exponential_isotropic", locs_c, NNarray_c, y_c, X_c, 
    //     &XSXl, &ySXl, &ySyl, &logdetl, &dXSXl, &dySXl, &dySyl,
    //     &dlogdetl, &ainfol, profbeta, grad_info);
    
    
    
    
    
    // XSX += XSXl;
    // ySX += ySXl;
    // ySy += ySyl;
    // logdet += logdetl;
    // dXSX += dXSXl;
    // dySX += dySXl;
    // dySy += dySyl;
    // dlogdet += dlogdetl;
    // ainfo += ainfol;

    

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

    // printf("dlogdet\n");
    // dlogdet.print();
    // printf("dySy\n");
    // dySy.print();
    // printf("abeta\n");
    // abeta.print();
    // printf("dySX\n");
    // dySX.print();
    // printf("ySX\n");
    // ySX.print();
    // printf("dbeta\n");
    // dbeta.print();
    // printf("dXSX\n");
    // dXSX.print();
    // for(int j=0; j<nparms; j++){
    //     (*grad)(j) = -(*grad)(j) * exp(covparms_c[j]); 
    // }
    // printf("likfun...done\n");

}

#define EXPIT(x) (exp(x)/(1+exp(x)))

double pen(arma::vec params, double vv) {
    double pen_nug = 0;
    if (params(2) != 0) {
        pen_nug = -.1*log(1+exp(-log(params(2))+log(0.001)));
    }
    double pen_var = -log(1+exp(params(0)/vv-6));
    return pen_nug + pen_var;
}

arma::vec dpen(arma::vec params, double vv) {
    arma::vec dpen_nug = arma::vec(3, fill::zeros);
    if (params(2) != 0) { 
        dpen_nug(2) = 0.1*EXPIT(-log(params(2))+log(.001))/params(2);
    }
    dpen_nug(0) = -1/vv * EXPIT(params(0)/vv-6);
    return dpen_nug;
}

arma::mat ddpen(arma::vec params, double vv) {
    arma::mat ddpen_nug = arma::mat(3, 3, fill::zeros);
    if (params[2] != 0) {
        double term = -log(params[2])+log(0.001);
        ddpen_nug(2,2) = -.1*EXPIT(term)/(1+exp(term))/(params[2]*params[2])-.1*EXPIT(term)/(params[2]*params[2]);
    }
    ddpen_nug(0,0) = -1/(vv * vv)*EXPIT(params[0]/vv-6) / (1 + exp(params[0]/vv - 6));
    return ddpen_nug;
}

// [[Rcpp::export]]
List fisher_scoring_gpu(
    NumericVector start_params, 
    NumericVector y,
    NumericMatrix X,
    NumericMatrix locs,
    NumericMatrix NNarray,
    double vv,
    bool silent,
    double convtol,
    int max_iter){
    
    // printf("fisher_scoring_gpu\n");

    bool profbeta = true;
    bool grad_info = true;

    int n = y.length();
    int m = NNarray.ncol();
    int p = X.ncol();
    int nparms = start_params.length();
    int dim = locs.ncol();

   

    arma::vec ll_a = arma::vec(1, fill::zeros);
    arma::vec grad_a = arma::vec(nparms, fill::zeros);
    arma::vec betahat_a = arma::vec(p, fill::zeros);
    arma::mat info_a = arma::mat(nparms, nparms, fill::zeros);
    arma::mat betainfo_a = arma::mat(p, p, fill::zeros);

    // double covparmsl[3] = { 0, 0, 0 };
    // covparmsl[0] = start_params[0];
    // covparmsl[1] = start_params[1];
    // covparmsl[2] = start_params[2];
    
    double* locsl = (double*)malloc(sizeof(double) * n * dim);
    double* NNarrayl = (double*) calloc(n * m, sizeof(double));//malloc(sizeof(double) * n * m);
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
            if (j <= i){
                NNarrayl[i * m + j] = NNarray(i, j);
            }
        }
    }
    double** loaded_data = load_data(locsl, NNarrayl, yl, Xl, n, m, dim, p, nparms);
    double* XSXf = (double*)calloc(p * p, sizeof(double));
    double* ySXf = (double*)calloc(p * p, sizeof(double));
    double ySy = 0;
    double logdet = 0;

    double* dXSXf = (double*)calloc(p * p * nparms, sizeof(double));
    double* dySXf = (double*)calloc(p * nparms, sizeof(double));
    double* dySyf = (double*)calloc(nparms, sizeof(double));
    double* dlogdetf = (double*)calloc(nparms, sizeof(double));
    double* ainfof = (double*)calloc(nparms * nparms, sizeof(double));

    
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

    arma::vec logparms = arma::vec(start_params.begin(),start_params.length());
    arma::mat locs_c = arma::mat(locs.begin(), locs.nrow(), locs.ncol());
    arma::mat NNarray_c = arma::mat(NNarray.begin(), NNarray.nrow(), NNarray.ncol());
    arma::vec y_c = arma::vec(y.begin(),y.length());
    arma::mat X_c = arma::mat(X.begin(),X.nrow(),X.ncol());

    // evaluate function at initial values
    likfun(&grad_a, &info_a, &betainfo_a, &ll_a, &betahat_a,
    locsl, NNarrayl, yl, Xl,
    dlogdet, dySy, ainfo, XSXl, ySXl, dXSXl,
    dySyl, dlogdetl, ainfol, dySXl,
    exp(logparms), locs_c, NNarray_c, y_c, X_c,
    XSXf, ySXf, dXSXf, dySXf, 
    dySyf, dlogdetf, ainfof, profbeta, grad_info, 
    n, m, p, nparms, dim, loaded_data);
    info_a = info_a % (exp(logparms) * exp(logparms).t()) - ddpen(exp(logparms), vv) % (exp(logparms) * exp(logparms).t());
    ll_a = -ll_a - pen(exp(logparms), vv);
    grad_a = (-grad_a) % exp(logparms) - dpen(exp(logparms), vv) % exp(logparms);
    // printf("%f\n", info_a(0,0));
    // printf("%f\n", info_a(2,2));
    // cout.precision(6);
    // cout.setf(ios::fixed);
    // info_a.raw_print(cout, "info:\n");

    // assign loglik, grad, and info
    double loglik_local = ll_a(0);
    arma::vec grad_local = grad_a;
    arma::mat info_local = info_a;

    // add a small amount of regularization
    double reg = 0.1 * min(diagvec(info_local));
    info_local(0,0) += reg;
    info_local(1,1) += reg;
    info_local(2,2) += reg;

    if (!silent) {
        printf("Iter %i:\n", 0);
        printf("pars =  %.4f, %.4f, %.4f\n", exp(logparms[0]), exp(logparms[1]), exp(logparms[2]));
        printf("loglik = %.6f\n", -ll_a(0));
        printf("grad = %.3f, %.3f, %.3f", -grad_a[0], -grad_a[1], -grad_a[2]);
        printf("\n\n");
    }

    int i = 1;
    double err = 10;
    double stepgrad = 0;
    while (i <= max_iter && err > convtol) {
        double ll0 = ll_a(0);
        double condition_number = 0;
        if (max(diagvec(info_local))/min(diagvec(info_local)) > 1e6){
            condition_number = max(diagvec(info_local))/min(diagvec(info_local));
        } else {
            vec eigval;
            mat eigvec;
            eig_sym(eigval, eigvec, info_local);
            double max_lambda = max(eigval);
            double min_lambda = min(eigval);
            condition_number = max_lambda / min_lambda;
        }
        
        double tol = 1e-10;
        if (condition_number > 1 / tol){
            vec eigval;
            mat eigvec;
            eig_sym(eigval, eigvec, info_local);
            double max_lambda = max(eigval);
            double min_lambda = min(eigval);
            arma::vec ee_ratios = eigval / max_lambda;
            for (int k = 0; k < ee_ratios.n_elem; k++) {
                if (ee_ratios[k] < 1e-5) {
                    ee_ratios[k] = 1e-5;
                }
            }
            info_local = eigvec * arma::diagmat(ee_ratios) * eigvec.t();
        }

        // printf("det: %f\n", arma::det(info_a));
        // cout.precision(10);
        // cout.setf(ios::fixed);
        // info_local.raw_print(cout, "info:\n");
        // grad_local.raw_print(cout, "grad:\n");
        arma::vec step = -solve(info_local, grad_local);
        
        // step.raw_print(cout, "step:\n");
        if (mean(square(step)) > 1){
            if(!silent) { printf("##\n"); }
            step = step/sqrt(mean(square(step)));
        }

        arma::vec newlogparams = logparms + step;
        // cout.precision(6);
        // cout.setf(ios::fixed);
        // newlogparams.raw_print(cout, "newlogparams:\n");
        likfun(&grad_a, &info_a, &betainfo_a, &ll_a, &betahat_a,
        locsl, NNarrayl, yl, Xl,
        dlogdet, dySy, ainfo, XSXl, ySXl, dXSXl,
        dySyl, dlogdetl, ainfol, dySXl,
        exp(newlogparams), locs_c, NNarray_c, y_c, X_c,
        XSXf, ySXf, dXSXf, dySXf, 
        dySyf, dlogdetf, ainfof, profbeta, grad_info, 
        n, m, p, nparms, dim, loaded_data);

        info_a = info_a % (exp(newlogparams) * exp(newlogparams).t()) - ddpen(exp(newlogparams), vv) % (exp(newlogparams) * exp(newlogparams).t());
        ll_a = -ll_a - pen(exp(newlogparams), vv);
        grad_a = (-grad_a) % exp(newlogparams) - dpen(exp(newlogparams), vv) % exp(newlogparams);

        // printf("%f\n", ll_a(0));
        //////////////////////////////////
        // check whether negative likelihood decreased
        // take smaller stepsize
        if (ll_a(0) > ll0) {
            step = 0.25*step;
            newlogparams = logparms + step;

            likfun(&grad_a, &info_a, &betainfo_a, &ll_a, &betahat_a,
            locsl, NNarrayl, yl, Xl,
            dlogdet, dySy, ainfo, XSXl, ySXl, dXSXl,
            dySyl, dlogdetl, ainfol, dySXl,
            exp(newlogparams), locs_c, NNarray_c, y_c, X_c,
            XSXf, ySXf, dXSXf, dySXf, 
            dySyf, dlogdetf, ainfof, profbeta, grad_info, 
            n, m, p, nparms, dim, loaded_data);
            info_a = info_a % (exp(newlogparams) * exp(newlogparams).t()) - ddpen(exp(newlogparams), vv) % (exp(newlogparams) * exp(newlogparams).t());
            ll_a = -ll_a - pen(exp(newlogparams), vv);
            grad_a = (-grad_a) % exp(newlogparams) - dpen(exp(newlogparams), vv) % exp(newlogparams);
        }
        // printf("%f\n", ll_a(0));
        // check again, move along gradient
        if(ll_a(0) > ll0) {
            // info0 <- diag( rep(mean(diag(info)),nrow(info)) )
            arma::mat info0 = info0.ones(nparms, nparms) * mean(diagvec(info_local));
            // printf("det: %f\n", arma::det(info_a));
            step = -solve(info0,grad_local);
            newlogparams = logparms + step;
            
            likfun(&grad_a, &info_a, &betainfo_a, &ll_a, &betahat_a,
            locsl, NNarrayl, yl, Xl,
            dlogdet, dySy, ainfo, XSXl, ySXl, dXSXl,
            dySyl, dlogdetl, ainfol, dySXl,
            exp(newlogparams), locs_c, NNarray_c, y_c, X_c,
            XSXf, ySXf, dXSXf, dySXf, 
            dySyf, dlogdetf, ainfof, profbeta, grad_info, 
            n, m, p, nparms, dim, loaded_data);
            info_a = info_a % (exp(newlogparams) * exp(newlogparams).t()) - ddpen(exp(newlogparams), vv) % (exp(newlogparams) * exp(newlogparams).t());
            ll_a = -ll_a - pen(exp(newlogparams), vv);
            grad_a = (-grad_a) % exp(newlogparams) - dpen(exp(newlogparams), vv) % exp(newlogparams);
        }
        // printf("%f\n", ll_a(0));
        // check once move, take smaller step along gradient
        if(ll_a(0) > ll0){
            // info0 <- diag( rep(max(diag(info)),nrow(info)) )
            arma::mat info0 = info0.ones(nparms, nparms) * max(diagvec(info_local));
            // printf("det: %f\n", arma::det(info_a));
            step = -solve(info0,grad_local);
            newlogparams = logparms + step;
            
            likfun(&grad_a, &info_a, &betainfo_a, &ll_a, &betahat_a,
            locsl, NNarrayl, yl, Xl,
            dlogdet, dySy, ainfo, XSXl, ySXl, dXSXl,
            dySyl, dlogdetl, ainfol, dySXl,
            exp(newlogparams), locs_c, NNarray_c, y_c, X_c,
            XSXf, ySXf, dXSXf, dySXf, 
            dySyf, dlogdetf, ainfof, profbeta, grad_info, 
            n, m, p, nparms, dim, loaded_data);
            info_a = info_a % (exp(newlogparams) * exp(newlogparams).t()) - ddpen(exp(newlogparams), vv) % (exp(newlogparams) * exp(newlogparams).t());
            ll_a = -ll_a - pen(exp(newlogparams), vv);
            grad_a = (-grad_a) % exp(newlogparams) - dpen(exp(newlogparams), vv) % exp(newlogparams);
        }
        // printf("%f\n", ll_a(0));
        stepgrad = dot(step, grad_local);

        logparms = logparms + step;
        loglik_local = ll_a(0);
        grad_local = grad_a;
        info_local = info_a;

        // if (i == 1){
        //     step.print();
        //     grad_a.print();
        //     printf("stepgrad: %f\n", stepgrad);
        // }
        

        if (!silent) {
            printf("Iter %i:\n", i);
            printf("pars = %.4f, %.4f, %.4f\n", exp(logparms[0]), exp(logparms[1]), exp(logparms[2]));
            printf("loglik = %.6f\n", -loglik_local);
            printf("grad = %.3f, %.3f, %.3f\n", grad_local[0], grad_local[1], grad_local[2]);
            printf("step dot grad = %f\n",stepgrad);
            printf("\n");
        }

        if(fabs(stepgrad) < convtol /*|| no_decrease*/){
            break;
        }

        i++;
    }
    
    NumericVector ll(1);
    NumericVector grad( start_params.length() );
    NumericVector betahat( X.ncol() );
    NumericVector sebeta( X.ncol() );
    NumericVector tbeta( X.ncol() );
    NumericMatrix info( start_params.length(), start_params.length() );
    NumericMatrix betainfo( X.ncol(), X.ncol() );
    NumericMatrix betacov( X.ncol(), X.ncol() );
    NumericVector covparms = {logparms[0], logparms[1], logparms[2]};
    logparms = log(logparms);
    NumericVector logparms_ret = {logparms[0], logparms[1], logparms[2]};

    // free_data(loaded_data);

    ll[0] = loglik_local;
    grad[0] = grad_local(0);
    grad[1] = grad_local(1);
    grad[2] = grad_local(2);

    
    arma::mat betacov_a = betainfo_a.i();
    arma::vec sebeta_a = sqrt(diagvec(betacov_a));
    arma::vec tbeta_a = betahat_a / sebeta_a;

    // printf("betainfo_a\n");
    // betainfo_a.print();
    // printf("betacov_a\n");
    // betacov_a.print();
    // printf("sebeta_a\n");
    // sebeta_a.print();
    // printf("tbeta_a\n");
    // tbeta_a.print();

    for (int i = 0; i < p; i++) {
        betahat[i] = betahat_a[i];
        sebeta[i] = sebeta_a[i];
        tbeta[i] = tbeta_a[i];
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            betainfo(i,j)= betainfo_a(i,j);
            betacov(i,j)= betacov_a(i,j);
        }
    }

    for (int i = 0; i < nparms; i++) {
        for (int j = 0; j < nparms; j++) {
            info(i,j)= info_local(i,j);
        }
    }

    List ret = List::create( Named("covparms") = logparms_ret, Named("logparms") = covparms,
        Named("betahat") = betahat, Named("sebeta") = sebeta, Named("betacov") = betacov,
        Named("tbeta")=tbeta, Named("loglik")= ll, Named("no_decrease")=false,Named("grad")=grad,
        Named("info")=info, Named("conv")=false);
    // printf("fisher_scoring_gpu...done\n");
    return ret;

}

// [[Rcpp::export]]
NumericVector Linv_mult_gpu(NumericMatrix Linv, NumericVector z, IntegerMatrix NNarray) {
    int n = z.length();
    int m = NNarray.ncol();

    double* Linvl = (double*)malloc(sizeof(double) * n * m);
    double* NNarrayl = (double*) calloc(n * m, sizeof(double));
    double* zl = (double*)malloc(sizeof(double) * n);
    
    for (int i = 0; i < n; i++) {
        zl[i] = z[i];
        for (int j = 0; j < m; j++) {
            Linvl[i * m + j] = Linv(i, j);
            if (j <= i){
                NNarrayl[i * m + j] = NNarray(i, j);
            }
        }
    }

    double* x = call_Linv_mult_gpu(Linvl, zl, NNarrayl, n, m);

    NumericVector ret(n);

    for (int i = 0; i < n; i++) {
        ret[i] = x[i];
    }

    return ret;
}
