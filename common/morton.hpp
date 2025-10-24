#pragma once

#include <cinttypes>

#ifdef __GNUC__
inline uint32_t clz(uint64_t n) {
  return (n == 0) ? 64 : __builtin_clzll(n); 
}

inline uint32_t clz(uint32_t n) {
  return (n == 0) ? 32 : __builtin_clzll(n); 
}
#else
// windows
inline uint64_t unpack64_msb(uint64_t ab, uint64_t shift = 32) {
  uint64_t mask = (uint64_t(1) << shift) - 1;
  return ((ab >> shift) & mask);
}

inline uint64_t unpack64_lsb(uint64_t ab, uint32_t shift = 32) {
  uint64_t mask = (uint64_t(1) << shift) - 1;
  return (ab & mask);
}

inline uint32_t clz(uint32_t n) {
  uint32_t index;
  uint8_t isNonzero = _BitScanReverse((unsigned long *)&index, n);
  return uint32_t(isNonzero ? 31 - index : 32);
}

inline uint32_t clz(uint64_t n) {
  uint32_t llz = clz(uint32_t(unpack64_msb(n)));
  uint32_t rlz = clz(uint32_t(unpack64_lsb(n)));

  return llz + (llz == 32) * rlz;
}
#endif

namespace morton {

inline uint64_t expand2(const uint64_t a) {
  uint64_t b = a;

  b = (b | b << 32) & 0x00000000FFFFFFFF;
  b = (b | b << 16) & 0x0000FFFF0000FFFF;
  b = (b | b <<  8) & 0x00FF00FF00FF00FF;
  b = (b | b <<  4) & 0x0F0F0F0F0F0F0F0F;
  b = (b | b <<  2) & 0x3333333333333333;
  b = (b | b <<  1) & 0x5555555555555555;

  return b;
}

inline uint64_t expand3(const uint64_t a) {
  uint64_t b = a;

  b = (b * 0x0000000100000001) & 0x001F00000000FFFF;
  b = (b * 0x0000000000010001) & 0x001F0000FF0000FF;
  b = (b * 0x0000000000000101) & 0x100F00F00F00F00F;
  b = (b * 0x0000000000000011) & 0x10C30C30C30C30C3;
  b = (b * 0x0000000000000005) & 0x1249249249249249;

  return b;
}

inline uint64_t encode(float v0, float v1) {
  constexpr uint64_t bits_per_dimension = 32;
  constexpr uint64_t divisions_per_dimension = uint64_t(1) << bits_per_dimension;

  constexpr float scale = float(divisions_per_dimension - 1);

  uint64_t x = uint64_t(v0 * scale);
  uint64_t y = uint64_t(v1 * scale);

  return (expand2(y) << 1) + expand2(x);
} 

inline uint64_t encode(float v0, float v1, float v2) {
  constexpr uint64_t bits_per_dimension = 21;
  constexpr uint64_t divisions_per_dimension = uint64_t(1) << bits_per_dimension;

  constexpr float scale = float(divisions_per_dimension - 1);

  uint64_t x = uint64_t(v0 * scale);
  uint64_t y = uint64_t(v1 * scale);
  uint64_t z = uint64_t(v2 * scale);

  return (expand3(z) << 2) + (expand3(y) << 1) + expand3(x);
} 

}  // namespace morton
