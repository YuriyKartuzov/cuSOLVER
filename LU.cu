#include "LU.h"
using namespace std;
using namespace chrono;

// gpuLU
double gpuLU(int N, bool debug) {
	steady_clock::time_point timeStart, timeEnd;// variables for timing
	cublasStatus_t stat;
	cudaError cudaStatus;
	cusolverStatus_t cusolverStatus;
	cusolverDnHandle_t handle;

	// Host Variables
	double *A, *B1, *B; 

	// Device variables
	double *d_A, *d_B, *d_Work;
	int * d_pivot, *d_info, Lwork;
	int info_gpu = 0;
	
	// PART 1 ---------------------------------------------------------------------------------------------
	// allocate memory, generate random numbers for the matrix
	A = (double *)malloc(N*N * sizeof(double));
	B = (double *)malloc(N * sizeof(double));
	B1 = (double *)malloc(N * sizeof(double));
	for (int i = 0; i<N*N; i++) A[i] = rand() / (double)RAND_MAX;   // Randomize A
	for (int i = 0; i<N; i++) B[i] = 0.0;                           // initialize B
	for (int i = 0; i<N; i++) B1[i] = 1.0;                          // Initialize B1
	double al = 1.0, bet = 0.0;                                     // setting coefficientimeStart cblas_dgemv function
	int incx = 1, incy = 1;

	// TIMER start
	timeStart = steady_clock::now();

	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, al, A, N, B1, incx, bet, B, incy); // multiply B=A*B1

	// TIMER end															 
	auto cblasRUN = steady_clock::now() - timeStart;
	
	// PART 2 ---------------------------------------------------------------------------------------------
	// prepare memory on the device
	cudaStatus = cudaGetDevice(0);
	cusolverStatus = cusolverDnCreate(&handle);
	cudaStatus = cudaMalloc((void **)& d_A, N * N * sizeof(double));
	cudaStatus = cudaMalloc((void **)& d_B, N * sizeof(double));
	cudaStatus = cudaMalloc((void **)& d_pivot, N * sizeof(int));
	cudaStatus = cudaMalloc((void **)& d_info, sizeof(int));

	cudaStatus = cudaMemcpy(d_A, A, N*N * sizeof(double), cudaMemcpyHostToDevice); // copy d_A <-A
	cudaStatus = cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice);   // copy d_B <-B

	// compute buffer size and prep memory
	cusolverStatus = cusolverDnDgetrf_bufferSize(handle, N, N, d_A, N, &Lwork);
	cudaStatus = cudaMalloc((void **)& d_Work, Lwork * sizeof(double));

	// START timer
	timeStart = steady_clock::now();

	// MAIN FUNCTIONS
	cusolverStatus = cusolverDnDgetrf(handle, N, N, d_A, N, d_Work, d_pivot, d_info);
	cusolverStatus = cusolverDnDgetrs(handle, CUBLAS_OP_N, N, 1, d_A, N, d_pivot, d_B, N, d_info);
	cudaStatus = cudaDeviceSynchronize();

	// END timer
	auto cusolverRUN = steady_clock::now() - timeStart;

	// Main output
	if (debug) {
		cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost); // d_info -> info_gpu
		printf("\n  Run successfull: ", info_gpu);
		cudaStatus = cudaMemcpy(B1, d_B, N * sizeof(double), cudaMemcpyDeviceToHost);    // d_B ->B1
		printf(" Solution : ");
		for (int i = 0; i < 5; i++) printf("%g, ", B1[i]);
		printf(" ... "); // print first componentimeStart of the solution
		printf("\n");
	}

	// Garbage Collection
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_pivot);
	cudaFree(d_info);
	cudaFree(d_Work);
	free(A); free(B); free(B1);
	cusolverStatus = cusolverDnDestroy(handle);
	//cudaDeviceReset();

	//system("pause"); // needed to executable
	double returnVal = duration_cast<milliseconds>(cusolverRUN).count();
	return returnVal;
}






