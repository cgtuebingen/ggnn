/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.

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

#ifndef INCLUDE_GGNN_SELECT_CUDA_KNN_WRS_SELECT_LAYER_CUH_
#define INCLUDE_GGNN_SELECT_CUDA_KNN_WRS_SELECT_LAYER_CUH_

#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include <gflags/gflags.h>
#include <cub/cub.cuh>

#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename T>
__global__ void
select(const T kernel) {
  kernel();
}

/*
 * Selection of K Points per B for Layers.
 */
template <typename ValueT, typename KeyT, int BLOCK_DIM_X, int Sglob>
struct WRSSelectionKernel {
  static constexpr int ITEMS_PER_THREAD = (2 * Sglob - 1) / BLOCK_DIM_X + 1;
  typedef cub::BlockRadixSort<ValueT, BLOCK_DIM_X, ITEMS_PER_THREAD, KeyT>
      BlockRadixSort;

  void launch(const cudaStream_t stream = 0) {
    VLOG(2) << "SelectionKernel -- B: " << B << " | B_offset: " << B_offset;
    select<<<B, BLOCK_DIM_X, 0, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    const int b = B_offset + blockIdx.x;

    const int S_current = S + int(b < S_offset);
    const int start = b*S + min(b, S_offset);

    ValueT keys[ITEMS_PER_THREAD];
    KeyT values[ITEMS_PER_THREAD];

    for (int item = 0; item < ITEMS_PER_THREAD; item++) {
      const int i = item * BLOCK_DIM_X + threadIdx.x;
      if (i < S_current) {
        const KeyT n = start + i;
        const float e =
            (-1 * logf(d_rng[n])) /
            // the top merge kernel is configured to output the matching values for the current layer
            // otherwise, we would need to translate n to the bottom layer
            (d_nn1_dist_buffer[n]
             + std::numeric_limits<float>::epsilon());
        keys[item] = e;
        values[item] = n;
      } else { //FIXME: if this happens, the following sym query will fail
        keys[item] = -1.f;
        values[item] = -1;
      }
    }

    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, values);

    __syncthreads();

    // block index / growth ==> index of the upper segment
    const int s_segment = b / c_G;
    // b % c_G ==> n-th segment contributing to the upper segment
    const int sg_segment = b - s_segment * c_G;

    // number of points contributed by the current block
    const int SG_current = SG + int(sg_segment < SG_offset);

    const int s_offset = s_segment * Sglob + sg_segment*SG + min(sg_segment, SG_offset);

    for (int item = 0; item < ITEMS_PER_THREAD; item++) {
      const int s = threadIdx.x + item * BLOCK_DIM_X;
      if (s < SG_current) {
        const KeyT n = values[item];

        d_selection[s_offset + s] = n;
        d_translation[s_offset + s] = (!layer) ? n : d_translation_layer[n];
      }
    }
  }

  int B;  // number of blocks to work on
  int B_offset;

  int S; // segment/block size in current layer
  int S_offset; // number of blocks with S+1 elements = remainder in division by block size (can only be non-zero for the base-layer)

  int SG; // S/G = number of points contributed from current segment to upper segment
  int SG_offset; // S%G = number of segments which contribute an additional point to the upper segment

  int layer;

  const KeyT* d_translation_layer;
  const float* d_nn1_dist_buffer;
  const float* d_rng;

  KeyT* d_selection;
  KeyT* d_translation;
};

#endif  // INCLUDE_GGNN_SELECT_CUDA_KNN_WRS_SELECT_LAYER_CUH_
