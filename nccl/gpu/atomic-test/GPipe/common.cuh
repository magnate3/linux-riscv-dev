#ifndef COMMON
#define COMMON

#include <stdio.h>

#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#define ROUND_UP_DIVISION(N, S) (((N) + (S) - 1) / (S))

#ifndef DEBUG
#define dbg_printf(...)
#else
#define dbg_printf(...) do { printf(__VA_ARGS__); } while (0)
#endif

#define CUDA_RESULT_CHECK(f) do {                                                                  \
    CUresult e = f;                                                                      \
    if (e != CUDA_SUCCESS) {                                                                 \
        printf("Cuda failure %s:%d: %s\n", __FILE__, __LINE__, _cudaGetErrorEnum(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

__device__ inline int GetThreadNum()
{
	return threadIdx.x;
}

__device__ inline bool isMainThread()
{
	return GetThreadNum() == 0;
}

/**
* @param target the number you want the factor to be close to
* @param number the number you want the result to be a factor of
*/
__host__ int getClosestFactor(int target, int number) {
	for (int i = 0; i < number; i++) {
		if (number % (target + i) == 0) {
			return target + i;
		} else if (number % (target - i) == 0) {
			return target - i;
		}
	}
	return number;
}

#endif
