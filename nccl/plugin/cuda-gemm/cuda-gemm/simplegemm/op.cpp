// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ATen/ATen.h" // @manual
#include "torch/extension.h" // @manual

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

void run_gemm(void* A, void* B, void* C, int M, int N, int K);
void run_pingpong(void* A, void* B, void* C, int M, int N, int K);
void run_stmatrix_gemm(void* A, void* B, void* C, int M, int N, int K);

at::Tensor gemm(at::Tensor a, at::Tensor b) {
  // a (m x k), b (k x n)
  auto c = a.new_empty({b.size(1), a.size(0)}).transpose(0, 1);
  run_gemm(
      a.data_ptr(),
      b.data_ptr(),
      c.data_ptr(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

at::Tensor pingpong(at::Tensor a, at::Tensor b) {
  // a (m x k), b (k x n)
  auto c = a.new_empty({b.size(1), a.size(0)}).transpose(0, 1);
  run_pingpong(
      a.data_ptr(),
      b.data_ptr(),
      c.data_ptr(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

at::Tensor stmatrix_gemm(at::Tensor a, at::Tensor b) {
  // a (m x k), b (k x n)
  auto c = a.new_empty({b.size(1), a.size(0)}).transpose(0, 1);
  run_stmatrix_gemm(
      a.data_ptr(),
      b.data_ptr(),
      c.data_ptr(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

TORCH_LIBRARY(gemm, m) {
  m.def("gemm", &gemm);
  m.def("pingpong", &pingpong);
  m.def("stmatrix_gemm", &stmatrix_gemm);
}
