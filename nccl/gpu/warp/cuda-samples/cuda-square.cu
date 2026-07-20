#include <stdio.h>
#include<cuda.h>

#define N 10

__global__ void print_square() {
    printf("square of %d = %d\n", threadIdx.x, threadIdx.x*threadIdx.x);
}

int main() {

    print_square<<<1, N>>>();

    cudaDeviceSynchronize();

    return 0;
}