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

#ifndef INCLUDE_GGNN_MERGE_CUDA_KNN_TOP_MERGE_LAYER_CUH_
#define INCLUDE_GGNN_MERGE_CUDA_KNN_TOP_MERGE_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_k_best_list.cuh"
#include "ggnn/utils/cuda_knn_distance.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename T>
__global__ void
top(const T kernel) {
  kernel();
}

template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int D, int K, int BLOCK_DIM_X,
          typename BaseT = ValueT, typename BAddrT = int32_t,
          typename GAddrT = int32_t>
struct TopMergeKernel {
  static constexpr int K_BEST = K;

  // this allows for loop unrolling
  static constexpr int ITERATIONS_FOR_K = (K+BLOCK_DIM_X-1)/BLOCK_DIM_X;

  void launch(const cudaStream_t stream = 0) {
    VLOG(1) << "TopMergeKernel -- Layer: " << layer << " | N: " << N << " [" << N_offset << " " << N_offset+N << "]\n";

    top<<<N, BLOCK_DIM_X, 0, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    typedef Distance<measure, ValueT, KeyT, D, BLOCK_DIM_X, BaseT, BAddrT> Distance;
    typedef KBestList<ValueT, KeyT, K, BLOCK_DIM_X> KBestList;

    const int n = N_offset + blockIdx.x;
    const int m = (!layer) ? n : d_translation[n];

    Distance distCalc(d_base, m);
    KBestList best;

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

    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        const GAddrT addr = static_cast<GAddrT>(n) * K + k;
        d_graph[addr] = best.ids[k];
      }
    }
    if (!threadIdx.x) {
      if (measure == Euclidean) {
        d_nn1_dist_buffer[n] = sqrt(best.dists[1]);
      }
      else if (measure == Cosine) {
        d_nn1_dist_buffer[n] = best.dists[1];
      }
    }
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

#endif  // INCLUDE_GGNN_MERGE_CUDA_KNN_TOP_MERGE_LAYER_CUH_
