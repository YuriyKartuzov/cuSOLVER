#pragma once
#include <chrono>
#include "gsl\gsl_cblas.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <string>

double gpuLU(int N, bool debug = false);
