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

#ifndef INCLUDE_GGNN_SYM_QUERY_LAYER_CUH
#define INCLUDE_GGNN_SYM_QUERY_LAYER_CUH

#include <ggnn/base/def.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void sym(const T kernel);

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
struct SymQueryKernel {
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;

  static constexpr uint32_t MAX_PER_PATH_ITERATIONS = 20;
  static constexpr uint32_t CACHE_SIZE = 128;
  static constexpr uint32_t MIN_PRIOQ_SIZE = 16;

  void launch(const uint32_t N, const cudaStream_t stream = 0)
  {
    VLOG(1) << "SymQueryKernel -- N: " << N;
    uint32_t sm_size = CACHE_SIZE * sizeof(KeyT) + sorted_size * sizeof(ValueT);

    CHECK_LT(sorted_size, CACHE_SIZE);
    CHECK_LE(D, BLOCK_DIM_X * DIST_ITEMS_PER_THREAD);

    sym<<<N, BLOCK_DIM_X, sm_size, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const;

  const uint32_t D;
  const DistanceMeasure measure;
  const uint32_t KBuild;
  // best size is KF = KBuild/2
  const uint32_t sorted_size = std::max(CACHE_SIZE < 512U ? 64U : 32U,
                                        next_multiple<uint32_t, 32>(KBuild / 2 + MIN_PRIOQ_SIZE));

  const BaseT* d_base;        // [N0,D]
  const KeyT* d_graph;        // [N,K]
  const KeyT* d_translation;  // [N] or nullptr if on base layer

  const float* d_nn1_stats;

  const float tau_build;

  KeyT* d_sym_buffer;      // [N,KF]
  uint32_t* d_sym_atomic;  // [N]
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_SYM_QUERY_LAYER_CUH
