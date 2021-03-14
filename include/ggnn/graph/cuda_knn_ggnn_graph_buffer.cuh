/* Copyright 2021 ComputerGraphics Tuebingen. All Rights Reserved.
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
//

#ifndef INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_BUFFER_CUH_
#define INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_BUFFER_CUH_

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "ggnn/utils/cuda_knn_utils.cuh"

/**
 * GGNN graph buffer data
 * auxiliary data needed for graph construction once per GPU
 *
 * @param KeyT datatype of dataset indices
 * @param ValueT distance value type
 */
template <typename KeyT, typename ValueT>
struct GGNNGraphBuffer {
  /// distance to nearest known neighbor per point
  ValueT* d_nn1_dist_buffer;

  //BUFFER
  KeyT* d_graph_buffer;
  KeyT* d_sym_buffer;

  float* d_rng;

  int* d_sym_atomic;
  int* d_statistics;

  // cub buffer
  void* d_temp_storage_sum;
  void* d_temp_storage_max;

  size_t temp_storage_bytes_sum{0};
  size_t temp_storage_bytes_max{0};

  char* d_memory;

  GGNNGraphBuffer(const int N, const int K, const int KF) {
    // just to make sure that everything is sufficiently aligned
    auto align8 = [](size_t size) -> size_t {return ((size+7)/8)*8;};

    const size_t graph_buffer_size = align8(static_cast<size_t>(N) * K * sizeof(KeyT));
    const size_t sym_buffer_size = align8(static_cast<size_t>(N) * KF * sizeof(KeyT));
    const size_t rng_size = align8(static_cast<size_t>(N) * sizeof(float));
    const size_t sym_atomic_size = align8(static_cast<size_t>(N) * sizeof(int));
    const size_t sym_statistics_size = align8(static_cast<size_t>(N) * sizeof(int));
    const size_t nn1_dist_buffer_size = align8(N * sizeof(ValueT));

    // stats
    {
      ValueT* d_nn1_stats_unused, *d_nn1_dist_buffer_unused;

      CHECK_CUDA(cudaMalloc(&d_nn1_stats_unused, nn1_dist_buffer_size+2*sizeof(ValueT)));
      d_nn1_dist_buffer_unused = d_nn1_stats_unused+2;

      cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_sum,
          d_nn1_dist_buffer_unused, &d_nn1_stats_unused[0], N);
      cub::DeviceReduce::Max(nullptr, temp_storage_bytes_max,
          d_nn1_dist_buffer_unused, &d_nn1_stats_unused[1], N);

      temp_storage_bytes_sum = align8(temp_storage_bytes_sum);
      temp_storage_bytes_sum = align8(temp_storage_bytes_max);

      CHECK_CUDA(cudaFree(d_nn1_stats_unused));
    }

    // const size_t total_size = graph_buffer_size + sym_buffer_size + rng_size + sym_atomic_size + sym_statistics_size + nn1_dist_buffer_size + temp_storage_bytes_sum + temp_storage_bytes_max;

    // this will work as long as the construction code remains as is
    const size_t merge_size  = nn1_dist_buffer_size + graph_buffer_size;
    const size_t select_size = nn1_dist_buffer_size + rng_size;
    const size_t stats_size  = nn1_dist_buffer_size + temp_storage_bytes_sum + temp_storage_bytes_max;
    const size_t sym_size    = sym_buffer_size + sym_atomic_size + sym_statistics_size;

    const size_t overlapped_size = max(max(merge_size, select_size), max(stats_size, sym_size));

    VLOG(2) << "GGNNGraphBuffer(): allocating GPU memory... ("
            << overlapped_size/(1024.0f*1024.0f*1024.0f) << " GB total).\n";

    {
      size_t free, total;
      CHECK_CUDA(cudaMemGetInfo(&free, &total));
      CHECK_GE(free, overlapped_size) << "out of memory.";
    }

    CHECK_CUDA(cudaMalloc(&d_memory, overlapped_size));

    d_nn1_dist_buffer  = reinterpret_cast<ValueT*>(d_memory);
    d_graph_buffer     = reinterpret_cast<KeyT*>(d_memory + nn1_dist_buffer_size);
    d_rng              = reinterpret_cast<float*>(d_memory + nn1_dist_buffer_size);
    d_temp_storage_sum = d_memory + nn1_dist_buffer_size;
    d_temp_storage_max = d_memory + nn1_dist_buffer_size + temp_storage_bytes_sum;
    d_sym_buffer       = reinterpret_cast<KeyT*>(d_memory);
    d_sym_atomic       = reinterpret_cast<int*>(d_memory + sym_buffer_size);
    d_statistics       = reinterpret_cast<int*>(d_memory + sym_buffer_size + sym_atomic_size);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());


    VLOG(2) << "GGNNGraphBuffer(): done.\n";
  }

  ~GGNNGraphBuffer() {
    CHECK_CUDA(cudaFree(d_memory));
  }
};

#endif  // INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_BUFFER_CUH_
