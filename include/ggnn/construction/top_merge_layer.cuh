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

#ifndef INCLUDE_GGNN_TOP_MERGE_LAYER_CUH
#define INCLUDE_GGNN_TOP_MERGE_LAYER_CUH

#include <ggnn/base/def.h>
#include <glog/logging.h>

#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void top(const T kernel);

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
struct TopMergeKernel {
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;

  void launch(const uint32_t N, const cudaStream_t stream = 0)
  {
    VLOG(1) << "TopMergeKernel -- Layer: " << layer << " | N: " << N << "\n";
    uint32_t sm_size = KBuild * sizeof(ValueT) + KBuild * sizeof(KeyT);

    CHECK_LE(D, BLOCK_DIM_X * DIST_ITEMS_PER_THREAD);

    top<<<N, BLOCK_DIM_X, sm_size, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const;

  const uint32_t D;
  const DistanceMeasure measure;
  const uint32_t KBuild;

  const BaseT* d_base;
  const KeyT* d_translation;

  KeyT* d_graph;
  ValueT* d_nn1_dist_buffer;

  const uint32_t S;
  const uint32_t S_offset;

  const uint32_t layer;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_TOP_MERGE_LAYER_CUH
