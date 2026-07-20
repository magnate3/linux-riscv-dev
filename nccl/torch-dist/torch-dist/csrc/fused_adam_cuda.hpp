#pragma once

#include <torch/extension.h>

/** This header is cpoied from APEX.
 *  Should expose them instead of using this one.
 */

/** Declarations copied from apex/csrc/amp_C_frontend.cpp */
void multi_tensor_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

/** Declarations copied from apex/contrib/csrc/optimizers/fused_adam_cuda.cpp */

void strided_check_finite(
  at::Tensor& overflow_flag,
  at::Tensor& p_copy,
  int stride,
  int clear_overflow_first);

void adam(
  at::Tensor& p,
  at::Tensor& p_copy,
  at::Tensor& m,
  at::Tensor& v,
  at::Tensor& g,
  float lr,
  float beta1,
  float beta2,
  float eps,
  float grad_scale,
  int step,
  int mode,
  int bias_correction,
  float decay);

void reversible_adam(
  at::Tensor& p,
  at::Tensor& p_copy,
  at::Tensor& m,
  at::Tensor& v,
  at::Tensor& g,
  float lr,
  float beta1,
  float beta2,
  float eps,
  float grad_scale,
  int step,
  int mode,
  int bias_correction,
  float decay);

void maybe_adam_undo(
  at::Tensor& overflow_flag,
  at::Tensor& p,
  at::Tensor& m,
  at::Tensor& v,
  at::Tensor& g,
  float lr,
  float beta1,
  float beta2,
  float eps,
  float grad_scale,
  int step,
  int mode,
  int bias_correction,
  float decay);

void maybe_cast(
  at::Tensor& overflow_flag,
  at::Tensor& p_in,
  at::Tensor& p_out);

void fused_adam_cuda_mt(
  int chunk_size,
  at::Tensor overflow_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float lr,
  float beta1,
  float beta2,
  float eps,
  float grad_scale,
  int step,
  int mode,
  int bias_correction,
  float decay);

void maybe_cast_cuda_mt(
  int chunk_size,
  at::Tensor overflow_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists);

