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

#ifndef CUDA_KNN_TOP_MERGE_LAYER_CUH_
#define CUDA_KNN_TOP_MERGE_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/cache/cuda_knn_multi_worked_dists_cache.cuh"
#include "ggnn/config.hpp"
#include "ggnn/cuda_knn_config.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename ValueT, typename KeyT, int D, int K, int BLOCK_DIM_X,
          typename BaseT = ValueT, typename BAddrT = int32_t,
          typename GAddrT = int32_t>
struct TopMergeKernel {
  static constexpr int K_BEST = K;

  void launch() {
    lprintf(1, "\nTopMergeKernel -- Layer: %d | N: %d [%d %d] \n", layer, N,
            N_offset, N_offset + N);

    launcher<<<N, BLOCK_DIM_X>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    typedef Distance<ValueT, KeyT, D, BLOCK_DIM_X, BaseT, BAddrT> Distance;
    typedef KBestList<ValueT, KeyT, K> KBestListTest;

    __shared__ union { typename Distance::TempStorage dist; } temp_storage;

    const int n = N_offset + blockIdx.x;
    const int m = (!layer) ? n : d_translation[n];

    Distance distCalc(&temp_storage.dist, d_base, m);
    KBestListTest best;

    const int S_plus_offset = S_offset * (S + 1);
    const int S_actual = (!layer && n < S_plus_offset) ? S + 1 : S;

    const int start =
        (layer || n < S_plus_offset)
            ? (n / S_actual) * S_actual
            : S_plus_offset + ((n - S_plus_offset) / S_actual) * S_actual;
    const int end = start + S_actual;

    for (int other_n = start; other_n < end; other_n++) {
      __syncthreads();
      const int other_m = (layer) ? d_translation[other_n] : other_n;

      if (m == other_m) continue;
      ValueT dist = distCalc.distance_synced(other_m);

      best.add_unique(dist, other_n);
    }

    if (threadIdx.x < K) {
      const GAddrT addr = static_cast<GAddrT>(n) * K + threadIdx.x;
      d_graph[addr] = best.ids[threadIdx.x];
    }
    if (!threadIdx.x) d_nn1_dist_buffer[n] = sqrt(best.dists[1]);
  }

  int N_offset;
  int N;

  int S;
  int S_offset;

  int layer;

  const BaseT* d_base;
  const KeyT* d_translation;

  KeyT* d_graph;
  ValueT* d_nn1_dist_buffer;
};

#endif  // CUDA_KNN_TOP_MERGE_LAYER_CUH_
