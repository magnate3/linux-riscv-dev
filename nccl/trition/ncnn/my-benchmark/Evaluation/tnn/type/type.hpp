#pragma once

#include <cstdint>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_bf16.h>  // IWYU pragma: export
#include <cuda_fp16.h>  // IWYU pragma: export
#else
#include "type/fp16.hpp"
#endif

namespace tnn {
#if defined(USE_CUDA)
typedef __half fp16;
typedef __nv_bfloat16 bf16;
#endif
typedef float fp32;
typedef double fp64;

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<bf16> {
  static constexpr const char *name = "bf16";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp32;
};

template <>
struct TypeTraits<fp16> {
  static constexpr const char *name = "fp16";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp32;
};

template <>
struct TypeTraits<fp32> {
  static constexpr const char *name = "float32";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp64;
};

template <>
struct TypeTraits<fp64> {
  static constexpr const char *name = "float64";
  static const float epsilon;
  using ComputePrecision = fp64;
  using HigherPrecision = fp64;
};

enum class DType_t : uint32_t { BYTE, FP16, BF16, FP32, FP64, INT32_T, INT64_T, SIZE_T, UNKNOWN };

template <typename T>
constexpr DType_t dtype_of() {
  return DType_t::UNKNOWN;
}
template <>
constexpr DType_t dtype_of<bf16>() {
  return DType_t::BF16;
}
template <>
constexpr DType_t dtype_of<fp16>() {
  return DType_t::FP16;
}
template <>
constexpr DType_t dtype_of<float>() {
  return DType_t::FP32;
}
template <>
constexpr DType_t dtype_of<double>() {
  return DType_t::FP64;
}
template <>
constexpr DType_t dtype_of<int32_t>() {
  return DType_t::INT32_T;
}
template <>
constexpr DType_t dtype_of<int64_t>() {
  return DType_t::INT64_T;
}
template <>
constexpr DType_t dtype_of<size_t>() {
  return DType_t::SIZE_T;
}

enum class SBool : uint8_t { FALSE = 0, TRUE = 1 };

inline float dtype_eps(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP16:
      return TypeTraits<fp16>::epsilon;
    case DType_t::BF16:
      return TypeTraits<bf16>::epsilon;
    case DType_t::FP32:
      return TypeTraits<fp32>::epsilon;
    case DType_t::FP64:
      return TypeTraits<fp64>::epsilon;
    default:
      throw std::runtime_error("Unknown data type for dtype_eps");
  }
}

inline size_t get_dtype_size(DType_t dtype) {
  switch (dtype) {
    case DType_t::BYTE:
      return sizeof(uint8_t);
    case DType_t::FP16:
      return sizeof(fp16);
    case DType_t::BF16:
      return sizeof(bf16);
    case DType_t::FP32:
      return sizeof(fp32);
    case DType_t::FP64:
      return sizeof(fp64);
    case DType_t::INT32_T:
      return sizeof(int32_t);
    case DType_t::INT64_T:
      return sizeof(int64_t);
    case DType_t::SIZE_T:
      return sizeof(size_t);
    default:
      throw std::runtime_error("Unknown data type for get_dtype_size");
  }
}

inline std::string dtype_to_string(DType_t dtype) {
  switch (dtype) {
    case DType_t::BYTE:
      return "byte";
    case DType_t::FP16:
      return "fp16";
    case DType_t::BF16:
      return "bf16";
    case DType_t::FP32:
      return "fp32";
    case DType_t::FP64:
      return "fp64";
    case DType_t::INT32_T:
      return "int32";
    case DType_t::INT64_T:
      return "int64";
    case DType_t::SIZE_T:
      return "size_t";
    default:
      return "unknown";
  }
}

}  // namespace tnn