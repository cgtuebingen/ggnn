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

#ifndef INCLUDE_GGNN_QUERY_CUDA_KNN_BF_QUERY_LAYER_CUH_
#define INCLUDE_GGNN_QUERY_CUDA_KNN_BF_QUERY_LAYER_CUH_

#include <algorithm>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "ggnn/utils/cuda_knn_k_best_list.cuh"
#include "ggnn/utils/cuda_knn_distance.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename T>
__global__ void
bf_query(const T kernel) {
  kernel();
}

/**
 * query which loops through all points and can be used to create ground truth data
 */
template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int D, int KQuery, int BLOCK_DIM_X,
          typename BaseT = ValueT, typename BAddrT = int32_t, typename GAddrT = int32_t,
          bool WRITE_DISTS = false>
struct BruteForceQueryKernel {
  typedef Distance<measure, ValueT, KeyT, D, BLOCK_DIM_X, BaseT, BAddrT> Distance;
  typedef KBestList<ValueT, KeyT, KQuery, BLOCK_DIM_X> KBestList;

  static constexpr int ITERATIONS_FOR_K_QUERY = (KQuery+BLOCK_DIM_X-1)/BLOCK_DIM_X;

  void launch(const cudaStream_t stream = 0) {
    DLOG(INFO) << "BruteForceQueryKernel -- KQuery: " << KQuery;
    bf_query<<<N, BLOCK_DIM_X, 0, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    const KeyT n = N_offset + static_cast<int>(blockIdx.x);

    Distance distCalc(d_base, d_query, n);
    KBestList best;
    __syncthreads();

    for (KeyT i=0; i<N_base; ++i)
    {
      // fetch the entire base, one by one
      ValueT dist = distCalc.distance_synced(i);
      if (dist < best.worst()) // should be faster than checking all elements
        best.add_unique(dist, i);
    }
    __syncthreads();

    for (int i=0; i < ITERATIONS_FOR_K_QUERY; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < KQuery) {
        const GAddrT addr = static_cast<GAddrT>(n) * KQuery + k;
        d_query_results[addr] = best.ids[k];
        if (WRITE_DISTS)
          d_query_results_dists[addr] = best.dists[k];
      }
    }
  }

  const BaseT* d_base;        // [Nall,D]
  const BaseT* d_query;       // [Nq,D]

  KeyT* d_query_results;          // [Nq,KQuery]
  ValueT* d_query_results_dists;  // [Nq,KQuery]

  int N_base;    // number of base points
  int N;         // number of points to query for -> Nq
  int N_offset;  // gpu offset in N
};

#endif  // INCLUDE_GGNN_QUERY_CUDA_KNN_BF_QUERY_LAYER_CUH_
