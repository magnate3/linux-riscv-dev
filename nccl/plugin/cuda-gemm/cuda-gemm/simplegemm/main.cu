#include "pingpong.cu"
//#include "stmatrix.cu"

__global__ void testFill(bf16* X, int M, int N, int parity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int m_idx = idx % M;
  int n_idx = idx / M;
  if (m_idx >= M || n_idx >= N)
    return;
  if (parity < 0) {
    X[idx] = (m_idx == n_idx) ? 1.0 : 0.0;
  } else {
    X[idx] = idx;
  }

  // int v = (idx % 8 - 4);
  // //v = (v >= 0) ? v + 1 : v;
  // //X[idx] = (bf16)(v * parity);
  // X[idx] = (float)(clock() % 8) / 8.0 - 0.5;
}

cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16* A, bf16* B, bf16* C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(
      cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      M,
      N,
      K,
      &alpha,
      A,
      CUDA_R_16BF,
      K,
      B,
      CUDA_R_16BF,
      K,
      &beta,
      C,
      CUDA_R_16BF,
      M,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS error: %d\n", status);
    exit(EXIT_FAILURE);
  }
}

__global__ __launch_bounds__(
    1024) void naive_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * N) {
    int m_idx = idx % M;
    int n_idx = idx / M;
    float sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += __bfloat162float(A[m_idx * K + k]) *
          __bfloat162float(B[k + n_idx * K]);
    }
    C[m_idx + n_idx * M] = __float2bfloat16(sum);
  }
}

void run_naive_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
  naive_gemm<<<cdiv(M * N, 1024), 1024>>>(A, B, C, M, N, K);
}

template<typename Gen>
void randomize_matrix(Gen& generator, bf16 *hM, bf16 *dM, int N) {
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) {
    hM[i] = distribution(generator);
  }
  check(cudaMemcpy(dM, hM, sizeof(bf16) * N, cudaMemcpyHostToDevice));
}

void arange(bf16 *hM, bf16* dM, int M, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      hM[m * N + n] = m * N + n;
    }
  }
  check(cudaMemcpy(dM, hM, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
}

void identity(bf16 *hM, bf16* dM, int M, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      hM[m + n * M] = (m == n) ? 1.0f : 0.0f;
    }
  }
  check(cudaMemcpy(dM, hM, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
}

void print_matrix(bf16* hM, bf16* dM, int M, int N, bool rowmajor) {
  check(cudaMemcpy(hM, dM, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));
  auto strideM = rowmajor ? N : 1;
  auto strideN = rowmajor ? 1 : M;
  for (int i = 0; i < min(M, 128); i++) {
    for (int j = 0; j < min(N, 128); j++) {
      printf("  %6.2f", __bfloat162float(hM[i * strideM + j * strideN]));
    }
    printf(" ...\n");
  }
  printf("...\n\n");
}

int main() {
  // int m = 6 * 11 * 128;
  // int n = 6 * 12 * 128;
  // int k = 512;

  // m = k = 8;
  // n = 16;

  int m = 6 * 11 * 128;
  int n = 3 * 12 * 256;
  int k = 1024;

  m = 6 * 11 * 128;
  n = 6 * 12 * 128;
  k = 640;

  // m = 8 * 128;
  // n = 8 * 256;
  // k = 64;
  //m = n = k = 8192;
  //int max = 8192;
  int max = 16384;
  int numel = max * max;

  // Allocate matrices
  __nv_bfloat16* A;
  __nv_bfloat16* B;
  __nv_bfloat16* C;
  __nv_bfloat16* Cref;

  check(cudaMalloc((void**)&A, sizeof(bf16) * max * max));
  check(cudaMalloc((void**)&B, sizeof(bf16) * max * max));
  check(cudaMalloc((void**)&C, sizeof(bf16) * max * max));
  check(cudaMalloc((void**)&Cref, sizeof(bf16) * max * max));

  bf16* hM = (bf16*)malloc(sizeof(bf16) * numel);

  // Fill with test data.
  //testFill<<<cdiv(numel, 1024), 1024>>>(A, m, k, 1);
  //testFill<<<cdiv(numel, 1024), 1024>>>(B, k, n, -1);
  std::default_random_engine gen(1337);
  randomize_matrix(gen, hM, A, numel);
  randomize_matrix(gen, hM, B, numel);
  //arange(hM, A, m, k);
  //identity(hM, B, k, n);
  //randomize_matrix(gen, hM, C, numel);
  check(cudaMemset(C, 0, sizeof(bf16) * numel));
  check(cudaGetLastError());

  // Generate cuBLAS reference.
  cublasCreate(&cublas_handle);
  runCublasGemmBF16(m, n, k, A, B, Cref);

  // Run test kernel.
  printf("about to run gemm\n");
  run_pingpong(A, B, C, m, n, k);

  // Print a slab of matrix for sanity.
  printf("A:\n"); print_matrix(hM, A, m, k, true);
  printf("B:\n"); print_matrix(hM, B, k, n, false);
  printf("C:\n"); print_matrix(hM, C, m, n, false);
  printf("Cref:\n"); print_matrix(hM, Cref, m, n, false);

  // Test against cuBLAS reference.
  bf16* hostC = nullptr;
  bf16* hostCref = nullptr;
  if (true) {
    hostC = (bf16*)malloc(sizeof(bf16) * m * n);
    hostCref = (bf16*)malloc(sizeof(bf16) * m * n);

    check(cudaMemcpy(hostC, C, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(
        hostCref, Cref, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < m * n; i++) {
      float cv = __bfloat162float(hostC[i]);
      float crefv = __bfloat162float(hostCref[i]);
      if (std::abs(cv - crefv) > 1e-5) {
        fprintf(
            stderr,
            "Failed tolerance check at idx (%d, %d), C=%f, Cref=%f\n",
            i / n, i % n,
            cv,
            crefv);
        exit(EXIT_FAILURE);
      }
    }
  }

  auto benchmark = false;
  if (benchmark) {
    // Benchmark test kernel.
    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    int repeat_times = 1000;
    float ms = 0.0f;
    check(cudaEventRecord(start));
    for (int j = 0; j < repeat_times; j++) {
      run_pingpong(A, B, C, m, n, k);
    }
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(start));
    check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&ms, start, stop));

    long flops = 2ll * m * n * k * repeat_times;
    printf("TFLOPS: %.1f\n", flops / ms * 1e-9);
  }

  // Free resources.
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(Cref);
  free(hM);
  free(hostC);
  free(hostCref);
  return 0;
}
