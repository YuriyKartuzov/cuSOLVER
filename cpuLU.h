#pragma once
#include <chrono>
#include <random>

using namespace std;
using namespace chrono;

double cpuLU(int n);
void getLU(double* A, double* L, double* U, int n); //A is the matrix, L is lower return, U is upper return, n is demension
void solveLU(double* L, double* U, int n, double* B, double* X); //L is lower, U is upper, n is demension, B is know vector, X is unknown vector, LUx=b