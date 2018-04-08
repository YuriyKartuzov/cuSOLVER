#include "gsl\gsl_cblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>

using namespace std;
using namespace chrono;

double Cholesky(int N, bool debug) {
	steady_clock::time_point timeStart, timeEnd;// variables for timing
	double accum; // elapsed time variable
	double *A, *B, *B1; // declare arrays on the host
						// prepare memory on the host

	A = (double *)malloc(N*N * sizeof(double)); // NxN coeff . matrix
	B = (double *)malloc(N * sizeof(double)); // N- vector rhs B=A*B1
	B1 = (double *)malloc(N * sizeof(double)); // auxiliary N- vect .
	for (int i = 0; i < N*N; i++) A[i] = rand() / (double)RAND_MAX;
	for (int i = 0; i < N; i++) B[i] = 0.0;
	for (int i = 0; i < N; i++) B1[i] = 1.0; // N- vector of ones
	for (int i = 0; i < N; i++) {
		A[i*N + i] = A[i*N + i] + (double)N; // make A positive definite
		for (int j = 0; j < i; j++)
			A[i*N + j] = A[j*N + i]; // and symmetric
	}

	double al = 1.0, bet = 0.0; // constants for dgemv
	int incx = 1, incy = 1;
	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, al, A, N, B1, incx, bet, B, incy); // B=A*B1

	// GPU
	cudaError cudaStatus;
	cusolverStatus_t cusolverStatus;
	cusolverDnHandle_t handle; // device versions of

	double *d_A, *d_B, *Work; // matrix A, rhs B and worksp .
	int *d_info, Lwork; // device version of info , worksp . size
	int info_gpu = 0; // device info copied to host
	cudaStatus = cudaGetDevice(0);
	cusolverStatus = cusolverDnCreate(&handle); // create handle
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	// DEVICE housekeeping
	timeStart = steady_clock::now();

	cudaStatus = cudaMalloc((void **)& d_A, N*N * sizeof(double));
	cudaStatus = cudaMalloc((void **)& d_B, N * sizeof(double));
	cudaStatus = cudaMalloc((void **)& d_info, sizeof(int));
	cudaStatus = cudaMemcpy(d_A, A, N*N * sizeof(double), cudaMemcpyHostToDevice); // copy A->d_A
	cudaStatus = cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice); // copy B->d_B

	// COMPUTE workspace size and prepare workspace
	cusolverStatus = cusolverDnDpotrf_bufferSize(handle, uplo, N, d_A, N, &Lwork);
	cudaStatus = cudaMalloc((void **)& Work, Lwork * sizeof(double));

	cusolverStatus = cusolverDnDpotrf(handle, uplo, N, d_A, N, Work, Lwork, d_info);
	// solve d_A *X=d_B , where d_A is factorized by potrf function
	// d_B is overwritten by the solution
	cusolverStatus = cusolverDnDpotrs(handle, uplo, N, 1, d_A, N, d_B, N, d_info);
	cudaStatus = cudaDeviceSynchronize();

	auto CholeskyRUN = steady_clock::now() - timeStart;

	if (debug) {
		printf(" solution : ");
		printf(" Dpotrf + Dpotrs time : %lf sec .\n", CholeskyRUN); // pr.el. time
		cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost); // copy d_info -> info_gpu
		printf(" after Dpotrf + Dpotrs : info_gpu = %d\n", info_gpu);
		cudaStatus = cudaMemcpy(B, d_B, N * sizeof(double), cudaMemcpyDeviceToHost); // copy solution to host d_B ->B

		for (int i = 0; i < 5; i++) printf("%g, ", B[i]); // print
		printf(" ... "); // first components of the solution
		printf("\n");
		}

	// Housekeeping
	cudaStatus = cudaFree(d_A);
	cudaStatus = cudaFree(d_B);
	cudaStatus = cudaFree(d_info);
	cudaStatus = cudaFree(Work);
	cusolverStatus = cusolverDnDestroy(handle);
	cudaStatus = cudaDeviceReset();

	double returnVal = duration_cast<milliseconds>(CholeskyRUN).count();
	return returnVal;
}