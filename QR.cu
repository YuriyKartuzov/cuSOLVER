#include "QR.h"
using namespace std;
using namespace chrono;

double QR(int m, bool debug) {

	steady_clock::time_point timeStart, timeEnd;// variables for timing
	cusolverDnHandle_t cusolverH; // cusolver handle
	cublasHandle_t cublasH; // cublas handle
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS

	const int lda = m; // leading dimension of A
	const int ldb = m; // leading dimension of B
	const int nrhs = 1; // number of right hand sides
	// A - mxm coeff . matr ., B=A*B1 -right hand side , B1 - mxnrhs
	double *A, *B, *B1, *X; // - auxil .matrix , X - solution
	
	// HOST matrices and vectors
	A =  (double *)malloc(m * m * sizeof(double));
	B =  (double *)malloc(m *     sizeof(double));
	B1 = (double *)malloc(m *     sizeof(double));
	X =  (double *)malloc(m *     sizeof(double));

	// Initialize and randomize data on HOST
	for (int i = 0; i< m * m; i++) A[i] = rand() / (double)RAND_MAX;
	for (int i = 0; i< m; i++) B[i] = 0.0;;
	for (int i = 0; i< m; i++) B1[i] = 1.0;

	// Solve for B = A * B1
	cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1, A, m, B1, 1, 0.0, B, 1); 
	
	// DEVICE matrices and vectors and other stuff
	double *d_A, *d_B, *d_tau, *d_work;
	int * devInfo;      // device version of info
	int lwork = 0;      // workspace size
	int info_gpu = 0;   // device info copied to host
	const double one = 1;

	// create cusolver and cublas handles
	cusolver_status = cusolverDnCreate(&cusolverH);
	cublas_status   = cublasCreate(&cublasH);

	// prepare memory on the DEVICE
	cudaMalloc((void **)& d_A,     sizeof(double) * m * m);
	cudaMalloc((void **)& d_tau,   sizeof(double) * m);
	cudaMalloc((void **)& d_B,     sizeof(double) * m);
	cudaMalloc((void **)& devInfo, sizeof(int));

	// HOST to DEVICE copy.    A -> d_A and B -> d_B
	cudaMemcpy(d_A, A, sizeof(double) * m * m, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(double) * m,     cudaMemcpyHostToDevice);

	 // compute buffer size for geqrf and prepare workspace on device
	cusolver_status = cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, m, &lwork);
	cudaMalloc((void **)& d_work, sizeof(double) * lwork);

	// START timer
	timeStart = steady_clock::now();
	
	// QR factorization for d_A ; R stored in upper triangle of d_A 
	// elementary reflectors vectors stored in lower triangle of d_A
	// elementary reflectors scalars stored in d_tau
	cusolver_status = cusolverDnDgeqrf(cusolverH, m, m, d_A, m, d_tau, d_work, lwork, devInfo);
	cudaDeviceSynchronize();

	// STOP timer
	timeEnd = steady_clock::now();
	steady_clock::duration cusolverRUN = timeEnd - timeStart;

	// Copies devInfo -> info_gpu to check error code after ingeqrf function
	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); 
	
	// compute d_B = Q^T*B using ormqr function
	cusolver_status = cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, 
		m, d_tau, d_B, m, d_work, lwork, devInfo);
	cudaStat1 = cudaDeviceSynchronize();

	// devInfo -> info_gpu to check error code after ormqr function
	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); 
	
	// check error code of ormqr function
	//printf(" after ormqr : info_gpu = %d\n", info_gpu);
	// write the original system A*X=(Q*R)*X=B in the form
	// R*X=Q^T*B and solve the obtained triangular system
	cublas_status = cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, 
		CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, m, d_B, m);
	cudaStat1 = cudaDeviceSynchronize();

	// copy d_B -> X where X is a solution
	cudaStat1 = cudaMemcpy(X, d_B, sizeof(double) * m, cudaMemcpyDeviceToHost); 

	// Garbage collection
	cudaFree(d_A);
	cudaFree(d_tau);
	cudaFree(d_B);
	cudaFree(devInfo);
	cudaFree(d_work);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
	cudaDeviceReset();

	// Return
	double returnVal = duration_cast<milliseconds>(cusolverRUN).count();
	return returnVal;
}

/*

The function cusolverDnDgeqrf computes in double precision the QR factorization
in the form of A = Q * R

A is a m x n matrix
R is uppter tiangular
Q is orthogonal matrix 
	- represented as a product of elementary reflectors
	- real scalars are stored in array d_tau

cusolverDnDorgqr function computes the orthogonal matrix Q 


*/