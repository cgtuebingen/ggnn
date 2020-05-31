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

#ifndef CUDA_KNN_SYM_BUFFER_MERGE_LAYER_CUH_
#define CUDA_KNN_SYM_BUFFER_MERGE_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/cache/cuda_knn_cache_sym.cuh"
#include "ggnn/cache/cuda_knn_multi_worked_dists_cache.cuh"
#include "ggnn/config.hpp"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename ValueT, typename KeyT, int K, int KF, int BLOCK_DIM_X,
          typename GAddrT = int32_t>
struct SymBufferMergeKernel {
  static constexpr int C_N = BLOCK_DIM_X / KF;
  static constexpr int KL = K - KF;

  void launch() {
    lprintf(2, "SymBufferMergeKernel -- N: %d [%d %d] \n", N, N_offset,
            N_offset + N);
    dim3 block(KF, C_N);
    launcher<<<(N - 1) / C_N + 1, block>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    const int n = N_offset + blockIdx.x * C_N + threadIdx.y;
    const int kf = threadIdx.x;

    if (n >= N) return;

    __shared__ KeyT s_sym_buffer[C_N * KF];
    __shared__ KeyT s_graph_buffer[C_N * KF];
    __shared__ int s_pos[C_N];
    __shared__ bool s_found[C_N];

    const int tid = threadIdx.y * KF + threadIdx.x;
    //# load buffer
    s_sym_buffer[tid] = d_sym_buffer[static_cast<GAddrT>(n) * KF + kf];
    s_graph_buffer[tid] = d_graph[static_cast<GAddrT>(n) * K + KL + kf];

    if (tid < C_N) {
      s_pos[tid] = d_sym_atomic[N_offset + blockIdx.x * C_N + tid];
      s_found[tid] = false;
    }
    __syncthreads();

    for (int i = 0; i < KF; i++) {
      __syncthreads();

      const KeyT r_sym_buffer = s_sym_buffer[tid];
      const KeyT r_graph = s_graph_buffer[threadIdx.y * KF + i];
      if (r_graph == r_sym_buffer) s_found[threadIdx.y] = true;
      __syncthreads();
      if (s_pos[threadIdx.y] < KF) {
        //# if not found include
        if (!s_found[threadIdx.y] && !threadIdx.x) {
          s_sym_buffer[threadIdx.y * KF + s_pos[threadIdx.y]] = r_graph;
          s_pos[threadIdx.y]++;
        }
      }
      __syncthreads();
      if (tid < C_N) s_found[tid] = false;
    }

    __syncthreads();
    const GAddrT addr_graph = static_cast<GAddrT>(n) * K + K - kf - 1;

    const KeyT res = s_sym_buffer[tid];
    d_graph[addr_graph] = (res >= 0) ? res : n;
  }

  const KeyT* d_sym_buffer;  // [N, KF]
  const KeyT* d_sym_atomic;  // [N]
  KeyT* d_graph;             // [N, K]

  int N;  // number of points to work on
  int N_offset;
};

#endif  // CUDA_KNN_SYM_BUFFER_MERGE_LAYER_CUH_
