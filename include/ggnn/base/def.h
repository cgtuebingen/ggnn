/* Copyright 2025 ComputerGraphics Tuebingen. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch
// converted to GGNN library by: Lukas Ruppert, Deborah Kornwolf

#ifndef INCLUDE_GGNN_DEF_H
#define INCLUDE_GGNN_DEF_H

#include <bit>
#include <cstddef>
#include <cstdint>

namespace ggnn {

enum class DistanceMeasure : int {
  Euclidean = 0,
  Cosine = 1
};

inline float sizeInGB(const size_t bytes)
{
  return static_cast<float>(bytes / (1024UL * 1024UL)) / 1024.0f;
}

#if __cpp_lib_int_pow2 < 202002L
#include <type_traits>

template <typename T, std::enable_if_t<std::is_same_v<T, uint32_t>, int> = 0>
constexpr uint32_t bit_ceil(T v) noexcept
{
  // from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}
#else
using std::bit_ceil;
#endif

template <typename T, T base, std::enable_if_t<std::is_same_v<T, uint32_t>, int> = 0>
constexpr T next_multiple(T v) noexcept
{
  return v % base == 0 ? v : base * (v / base + 1);
};

// just to make sure that everything is sufficiently aligned
inline size_t align8(size_t size)
{
  return ((size + 7) / 8) * 8;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_DEF_H
