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

#ifndef INCLUDE_GGNN_WRS_SELECT_LAYER_CUH
#define INCLUDE_GGNN_WRS_SELECT_LAYER_CUH

#include <ggnn/base/def.h>
#include <glog/logging.h>

#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void select(const T kernel);

/*
 * Selection of K Points per B for Layers.
 */
template <typename KeyT, typename ValueT>
struct WRSSelectionKernel {
  static constexpr uint32_t BLOCK_DIM_X = 128;
  static constexpr uint32_t ITEMS_PER_THREAD = 2;

  void launch(const uint32_t B, const cudaStream_t stream = 0)
  {
    VLOG(2) << "SelectionKernel -- B: " << B;  // number of blocks to work on

    CHECK_LE(S + (S_offset > 0), ITEMS_PER_THREAD * BLOCK_DIM_X);
    CHECK_LE(SG + (SG_offset > 0), ITEMS_PER_THREAD * BLOCK_DIM_X);

    select<<<B, BLOCK_DIM_X, 0, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const;

  KeyT* d_selection;
  KeyT* d_translation;
  const KeyT* d_translation_layer;
  const float* d_nn1_dist_buffer;
  const float* d_rng;

  const uint32_t Sglob;     // segment/block size in upper layer (global segment size)
  const uint32_t S;         // segment/block size in current layer
  const uint32_t S_offset;  // number of blocks with S+1 elements (can only be > 0 for base layer)

  const uint32_t G;   // growth factor
  const uint32_t SG;  // S/G = number of points contributed per segment from lower to upper layer
  const uint32_t SG_offset;  // S%G = number of segments which contribute an additional point to the
                             // upper segment

  const uint32_t layer;  // bottom layer to select from
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_WRS_SELECT_LAYER_CUH
