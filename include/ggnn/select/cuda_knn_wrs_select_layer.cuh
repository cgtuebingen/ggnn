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
// Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

#ifndef CUDA_KNN_WRS_SELECT_LAYER_CUH_
#define CUDA_KNN_WRS_SELECT_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/config.hpp"
#include "ggnn/cuda_knn_config.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

/*
 * Selection of K Points per B for Layers.
 */
template <typename ValueT, typename KeyT, int BLOCK_DIM_X, int Sglob>
struct WRSSelectionKernel {
  static constexpr int ITEMS_PER_THREAD = (2 * Sglob - 1) / BLOCK_DIM_X + 1;
  typedef cub::BlockRadixSort<ValueT, BLOCK_DIM_X, ITEMS_PER_THREAD, KeyT>
      BlockRadixSort;

  void launch() {
    lprintf(2, "SelectionKernel: B: %d [%d %d] \n", B, B_offset, B_offset + B);
    launcher<<<B, BLOCK_DIM_X>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    const int b = B_offset + blockIdx.x;

    const int S_plus_offset = (!layer) ? S_offset * (S + 1) : 0;
    const int S_actual = (!layer && b < S_offset) ? S + 1 : S;

    const int start = (layer || b < S_offset)
                          ? b * S_actual
                          : S_plus_offset + (b - S_offset) * S_actual;

    ValueT keys[ITEMS_PER_THREAD];
    KeyT values[ITEMS_PER_THREAD];

    for (int item = 0; item < ITEMS_PER_THREAD; item++) {
      const int i = item * BLOCK_DIM_X + threadIdx.x;
      if (i < S_actual) {
        const KeyT n = start + i;
        const float e =
            (-1 * logf(d_rng[n])) /
            (d_nn1_dist_buffer[n] + std::numeric_limits<float>::epsilon());
        keys[item] = e;
        values[item] = n;
      } else {
        keys[item] = -1.f;
        values[item] = -1;
      }
    }

    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, values);

    __syncthreads();

    const int s_segment = b / c_G;
    const int sg_segment = b - s_segment * c_G;

    const bool is_sg_offset = sg_segment < SG_offset;
    const int SG_actual = (is_sg_offset) ? SG + 1 : SG;

    const int s_offset =
        s_segment * Sglob +
        ((is_sg_offset) ? sg_segment * SG_actual
                        : SG_offset * (SG + 1) + (sg_segment - SG_offset) * SG);

    for (int item = 0; item < ITEMS_PER_THREAD; item++) {
      const int s = threadIdx.x + item * BLOCK_DIM_X;
      if (s < SG_actual) {
        const KeyT n = values[item];

        d_selection[s_offset + s] = n;
        d_translation[s_offset + s] = (!layer) ? n : d_translation_layer[n];
      }
    }
  }

  int B;  // number of blocks to work on
  int B_offset;

  int S;
  int S_offset;

  int SG;
  int SG_offset;

  int layer;

  const KeyT* d_translation_layer;
  const float* d_nn1_dist_buffer;
  const float* d_rng;

  KeyT* d_selection;
  KeyT* d_translation;
};

#endif  // CUDA_KNN_WRS_SELECT_LAYER_CUH_
