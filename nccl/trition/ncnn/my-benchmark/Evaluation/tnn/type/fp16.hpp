#pragma once

#include <cstdint>
#include <cstring>

namespace tnn {

struct fp16 {
  uint16_t data;

  fp16() : data(0) {}
  explicit fp16(uint16_t d) : data(d) {}

  // Constructors from other types
  fp16(float f) : data(float_to_fp16(f)) {}
  fp16(double d) : data(float_to_fp16(static_cast<float>(d))) {}
  fp16(int i) : data(float_to_fp16(static_cast<float>(i))) {}
  fp16(size_t s) : data(float_to_fp16(static_cast<float>(s))) {}

  // Conversion operators
  // Non-explicit conversions to float/double so fp16 works with math functions like exp()
  operator float() const { return fp16_to_float(data); }
  operator double() const { return static_cast<double>(fp16_to_float(data)); }

  // Conversion to uint16_t for raw bit access (e.g., serialization, endian conversion)
  operator uint16_t() const { return data; }

  // Explicit conversion to size_t for indexing operations (e.g., embedding layers)
  explicit operator size_t() const { return static_cast<size_t>(fp16_to_float(data)); }

  static uint16_t float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>(((bits >> 23) & 0xff)) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;

    if (exp <= 0) {
      if (exp < -10) {
        return static_cast<uint16_t>(sign);
      }
      mantissa = (mantissa | 0x800000) >> (1 - exp);
      return static_cast<uint16_t>(sign | (mantissa >> 13));
    } else if (exp == 0xff - 127 + 15) {
      if (mantissa == 0) {
        return static_cast<uint16_t>(sign | 0x7c00);
      } else {
        return static_cast<uint16_t>(sign | 0x7c00 | (mantissa >> 13));
      }
    } else if (exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7c00);
    }

    return static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
  }

  static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;

    uint32_t bits;
    if (exp == 0) {
      if (mantissa == 0) {
        bits = sign;
      } else {
        exp = 1;
        while ((mantissa & 0x400) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x3ff;
        bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
      }
    } else if (exp == 31) {
      bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
      bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
  }
};

// Comparison operators
inline bool operator==(const fp16 &a, const fp16 &b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

inline bool operator!=(const fp16 &a, const fp16 &b) { return !(a == b); }

inline bool operator<(const fp16 &a, const fp16 &b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator>(const fp16 &a, const fp16 &b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator<=(const fp16 &a, const fp16 &b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>=(const fp16 &a, const fp16 &b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

// Arithmetic operators
inline fp16 operator+(const fp16 &a, const fp16 &b) {
  return fp16(static_cast<float>(a) + static_cast<float>(b));
}

inline fp16 operator-(const fp16 &a, const fp16 &b) {
  return fp16(static_cast<float>(a) - static_cast<float>(b));
}

inline fp16 operator*(const fp16 &a, const fp16 &b) {
  return fp16(static_cast<float>(a) * static_cast<float>(b));
}

inline fp16 operator/(const fp16 &a, const fp16 &b) {
  return fp16(static_cast<float>(a) / static_cast<float>(b));
}

// Compound assignment operators
inline fp16 &operator+=(fp16 &a, const fp16 &b) {
  a = a + b;
  return a;
}

inline fp16 &operator-=(fp16 &a, const fp16 &b) {
  a = a - b;
  return a;
}

inline fp16 &operator*=(fp16 &a, const fp16 &b) {
  a = a * b;
  return a;
}

inline fp16 &operator/=(fp16 &a, const fp16 &b) {
  a = a / b;
  return a;
}

// Unary operators
inline fp16 operator-(const fp16 &a) { return fp16(-static_cast<float>(a)); }

inline fp16 operator+(const fp16 &a) { return a; }

}  // namespace tnn