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

#ifndef INCLUDE_GGNN_SYM_BUFFER_MERGE_LAYER_CUH
#define INCLUDE_GGNN_SYM_BUFFER_MERGE_LAYER_CUH

#include <ggnn/base/def.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void sym_buffer_merge(const T kernel, const uint32_t N);

template <typename KeyT, typename ValueT>
struct SymBufferMergeKernel {
  static constexpr uint32_t BLOCK_DIM_X = 128;

  void launch(const uint32_t N, const cudaStream_t stream = 0)
  {
    VLOG(2) << "SymBufferMergeKernel -- N: " << N;
    dim3 block(KF, POINTS_PER_BLOCK);
    size_t sm_size = sizeof(KeyT) * POINTS_PER_BLOCK * KF * 2 + sizeof(bool) * POINTS_PER_BLOCK;
    sym_buffer_merge<<<(N - 1) / POINTS_PER_BLOCK + 1, block, sm_size, stream>>>((*this), N);
  }

  __device__ __forceinline__ void operator()(uint32_t N) const;
  const uint32_t KBuild;
  const uint32_t KF{KBuild / 2};

  const uint32_t POINTS_PER_BLOCK = BLOCK_DIM_X / KF;
  const uint32_t KL = KBuild - KF;

  const KeyT* d_sym_buffer;      // [N, KF]
  const uint32_t* d_sym_atomic;  // [N]
  KeyT* d_graph;                 // [N, K]
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_SYM_BUFFER_MERGE_LAYER_CUH
