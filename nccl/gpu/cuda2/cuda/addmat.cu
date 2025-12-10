#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void MatAdd(int tam, float *A, float *B, float *C){
   int i = threadIdx.x;
   int j = threadIdx.y;
   if(i < tam && j < tam)
       C[i*tam+j] = A[i*tam+j] + B[i*tam+j];
}

int main() {
  
   // thread x: id x
   // thread x,y: x + yDx
   // thread x,y,z: x + yDx + zDxDy

   int N = 2;
   float *h_A = (float*)malloc(N*N*sizeof(float));
   float *h_B = (float*)malloc(N*N*sizeof(float));
   float *h_C = (float*)malloc(N*N*sizeof(float));

   float *d_A, *d_B, *d_C;
   cudaMalloc((void**)&d_A, N*N*sizeof(float));
   cudaMalloc((void**)&d_B, N*N*sizeof(float));
   cudaMalloc((void**)&d_C, N*N*sizeof(float));

   for(int i=0; i<N; i++){
       for(int j=0; j<N; j++){
           h_A[i*N+j] = (i+j+1)*1;
           h_B[i*N+j] = (i+j+1)*2;
       }
  }

   cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

   int numBlocks = 1;
   dim3 threadsPerBlock(N,N);
   MatAdd<<<numBlocks, threadsPerBlock>>>(N,d_A,d_B,d_C);
   cudaDeviceSynchronize();

   cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

   for (int i = 0; i < N; ++i) {
       for (int j = 0; j < N; ++j) {
           printf("%2.2f ", h_C[i*N + j]);
       }
       printf("\n");
   }

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
   free(h_A);
   free(h_B);
   free(h_C);

   return 0;
}
