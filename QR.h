#pragma once
#include "gsl/gsl_cblas.h"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

double QR(int N, bool debug = false);
