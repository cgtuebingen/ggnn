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

#ifndef INCLUDE_GGNN_MERGE_LAYER_CUH
#define INCLUDE_GGNN_MERGE_LAYER_CUH

#include <ggnn/base/def.h>
#include <ggnn/base/graph_config.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include <glog/logging.h>

namespace ggnn {

template <typename T>
__global__ void merge(const T kernel);

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
struct MergeKernel {
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;

  static constexpr uint32_t MAX_ITERATIONS = 200;
  static constexpr uint32_t CACHE_SIZE = 256;
  static constexpr uint32_t MIN_PRIOQ_SIZE = 16;

  void launch(const uint32_t N, const cudaStream_t stream = 0)
  {
    CHECK_GT(layer_top, layer_btm);
    VLOG(1) << "MergeKernel -- Layer: " << layer_top << " -> " << layer_btm << " |  N: " << N
            << "\n";
    uint32_t sm_size = CACHE_SIZE * sizeof(KeyT) + SORTED_SIZE * sizeof(ValueT);
    CHECK_LT(SORTED_SIZE, CACHE_SIZE);
    CHECK_LE(D, BLOCK_DIM_X * DIST_ITEMS_PER_THREAD);

    merge<<<N, BLOCK_DIM_X, sm_size, stream>>>((*this));
  }

  // determine the start of the top-layer segment (always 0 for layer_top = L-1)
  __device__ __forceinline__ uint32_t get_top_seg_offset(const KeyT n) const;

  __device__ __forceinline__ void operator()() const;

  const uint32_t D;
  const DistanceMeasure measure;
  const uint32_t KBuild;
  const uint32_t SORTED_SIZE = std::max(CACHE_SIZE < 512U ? 64U : 32U,
                                        next_multiple<uint32_t, 32>(KBuild + 1 + MIN_PRIOQ_SIZE));
  const uint32_t S;

  const BaseT* d_base;        // [Nall,D]
  const KeyT* d_selection;    // [Sall]
  const KeyT* d_translation;  // [Nall]

  const KeyT* d_graph;   // [N,K]
  KeyT* d_graph_buffer;  // [N,K]

  const float* d_nn1_stats;  // [sum,max]
  float* d_nn1_dist_buffer;  // [N0]

  const uint32_t layer_top;  // layer to start from
  const uint32_t layer_btm;  // layer to merge

  const uint32_t G;          // growth factor
  const uint32_t S0;         // segment size on layer 0
  const uint32_t S0_offset;  // segment size offset on layer 0

  const std::array<uint32_t, GraphConfig::L> Ns_offsets;   // start position of graph layer
  const std::array<uint32_t, GraphConfig::L> STs_offsets;  // start position of translation layer

  const float tau_build;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_MERGE_LAYER_CUH
