#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void VecAdd(int *A, int *B, int *C, int N){
   int i = threadIdx.x;
   if(i < N){
       C[i] = A[i] + B[i];
   }
}

int main() {
  
   int N = 2;
   int *h_A = (int*)malloc(N*sizeof(int));
   int *h_B = (int*)malloc(N*sizeof(int));
   int *h_C = (int*)malloc(N*sizeof(int));
  
   int *d_A, *d_B, *d_C;
   cudaMalloc((void**)&d_A, N*sizeof(int));
   cudaMalloc((void**)&d_B, N*sizeof(int));
   cudaMalloc((void**)&d_C, N*sizeof(int));

   for(int i=0; i<N; i++){
       h_A[i] = i*1;
       h_B[i] = i*2;
   }

   cudaMemcpy(d_A, h_A, N*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, N*sizeof(int), cudaMemcpyHostToDevice);

   dim3 blockDim(16,16);
   dim3 gridDim(1,1,1);

   VecAdd<<<1,N>>>(d_A, d_B, d_C, N);
   cudaDeviceSynchronize();

   cudaMemcpy(h_C, d_C, N*sizeof(int), cudaMemcpyDeviceToHost);

   for(int i=0; i<N; i++){
       cout << h_C[i] << " ";
   }

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
   free(h_A);
   free(h_B);
   free(h_C);
   return 0;
}
