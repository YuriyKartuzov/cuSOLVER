#include "cpuLU.h"
typedef std::chrono::high_resolution_clock Clock;

double cpuLU(int n)
{
	std::chrono::steady_clock::time_point timeStart, timeEnd;// variables for timing

	double* L = new double[n * n];
	double* U = new double[n * n];
	double* X = new double[n];
	double* A = new double[n * n];
	double* B = new double[n];

	//generate double values
	srand(-1);
	for (int i = 0; i < n; i++) {
		B[i] = rand() % 10000;
	}
	for (int i = 0; i < n* n; i++) {
		A[i] = rand() % 10000;
	}

	timeStart = Clock::now();
	getLU(A, L, U, n);
	solveLU(L, U, n, B, X);
	timeEnd = Clock::now();

	return double(duration_cast<milliseconds>(timeEnd - timeStart).count());
}

void getLU(double* A, double* L, double* U, int n)
{
	//pivot variable
	double pivot;

	//L is identity matrix at start
	//set values to 0
	for (int i = 0; i < n * n; i++) {
		L[i] = 0.0;
	}
	//set diagonal to 1
	for (int i = 0; i < n * n; i += n + 1) {
		L[i] = 1.0;
	}
	//U will be A at first and transfromed after
	for (int i = 0; i < n * n; i++) {
		U[i] = A[i];
	}

	//for every pivot
	for (int pivotN = 0; pivotN < n; pivotN++) {
		//modify rows below
		for (int row = pivotN + 1; row < n; row++) {
			//get pivot 
			pivot = U[row * n + pivotN] / U[pivotN * n + pivotN];
			//apply pivot to element and everything right of element
			for (int i = pivotN; i < n; i++) {
				U[row * n + i] -= pivot * U[pivotN * n + i];
			}
			//apply pivot to lower table
			L[row * n + pivotN] = pivot;
		}
	}
}

void solveLU(double * L, double * U, int n, double * B, double * X)
{
	//LUx=b
	double* y = new double[n];
	// Ly = b
	for (int pivotN = 0; pivotN < n; pivotN++) {
		double sum = 0;
		//for each element with 0
		for (int i = 0; i < pivotN; i++) {
			sum -= L[pivotN * n + i] * y[i];
		}
		y[pivotN] = sum + B[pivotN];
	}

	//Ux = y
	//first row exeption
	for (int i = n - 1; i >= 0; i--) {
		double sum = 0;
		//for each element with 0
		for (int j = n - 1; j >= i + 1; j--) {
			sum -= X[j];
		}
		sum += y[i];
		X[i] = sum / U[i * n + i];
	}
}