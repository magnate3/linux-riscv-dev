#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cassert>
#include <fstream>
#include <map>
#include "cuda_runtime.h"
using namespace std;
__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
{
#if (CUDART_VERSION >= 3000)
#else
#endif
    return cudaSuccess;
}
