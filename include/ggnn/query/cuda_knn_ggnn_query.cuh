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
//

#ifndef INCLUDE_GGNN_UTILS_CUDA_KNN_GGNN_QUERY_CUH_
#define INCLUDE_GGNN_UTILS_CUDA_KNN_GGNN_QUERY_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <limits>
#include <vector>

#include <cub/cub.cuh>
#include "ggnn/utils/cuda_knn_utils.cuh"

/**
 * GGNN graph data
 *
 * @param KeyT datatype of dataset indices
 * @param ValueT dists value type
 * @param BaseT base value type
 */

template <typename KeyT, typename ValueT, typename BaseT>
struct GGNNQuery {
  /// query vectors
  BaseT* d_query{nullptr};
  KeyT* d_query_result_ids{nullptr};
  ValueT* d_query_result_dists{nullptr};

  KeyT* d_query_result_ids_sorted{nullptr};
  ValueT* d_query_result_dists_sorted{nullptr};

  /// number of dataset vectors
  const int N_query;
  /// dimension of vectors in the dataset and query
  const int D;
  /// number of nearest neighbors per ground truth entry
  const int K_query;

  const int num_parts;

  // Sort buffer:
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  int num_items = 0;
  int num_segments = 0;

  int* d_offsets{nullptr};


  GGNNQuery(const int N_query, const int D, const int K_query, const int num_parts = 1) : N_query{N_query}, D{D}, K_query{K_query}, num_parts{num_parts} {
    CHECK_CUDA(cudaMalloc(
        &d_query, static_cast<size_t>(N_query) * D * sizeof(BaseT)));
    
    CHECK_CUDA(cudaMalloc(
        &d_query_result_ids, static_cast<size_t>(N_query) * K_query * num_parts * sizeof(KeyT)));
    CHECK_CUDA(cudaMalloc(
        &d_query_result_ids_sorted, static_cast<size_t>(N_query) * K_query * num_parts * sizeof(KeyT)));
    
    CHECK_CUDA(cudaMalloc(
        &d_query_result_dists, static_cast<size_t>(N_query) * K_query * num_parts * sizeof(ValueT)));
    CHECK_CUDA(cudaMalloc(
        &d_query_result_dists_sorted, static_cast<size_t>(N_query) * K_query * num_parts * sizeof(ValueT)));

    num_items = static_cast<size_t>(N_query) * K_query * num_parts;
    num_segments = N_query;

    if (num_parts > 1) {
      const size_t segments_size = (num_segments+1)*sizeof(int);
      int* h_offsets = (int*) malloc(segments_size);
      CHECK_CUDA(cudaMalloc(&d_offsets, segments_size));

      for (int i = 0; i < (num_segments + 1); i++) {
        h_offsets[i] = i * K_query * num_parts;
      }
      CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, segments_size, cudaMemcpyHostToDevice));
      free(h_offsets);

      cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes,
      d_query_result_dists, d_query_result_dists_sorted, d_query_result_ids, d_query_result_ids_sorted,
      num_items, num_segments, d_offsets, d_offsets + 1);

      CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }
  }

  ~GGNNQuery() {
    cudaFree(d_query);
    cudaFree(d_query_result_ids);
    cudaFree(d_query_result_dists);
  }

  GGNNQuery(const GGNNQuery&) = delete;
  GGNNQuery(GGNNQuery&&) = delete;
  GGNNQuery& operator=(const GGNNQuery&) = delete;
  GGNNQuery& operator=(GGNNQuery&&) = delete;

  void loadQueriesAsync(const BaseT* h_query, const cudaStream_t stream){
    CHECK_CUDA(cudaMemcpyAsync(d_query,h_query,
                static_cast<size_t>(N_query) * D * sizeof(BaseT),
                cudaMemcpyHostToDevice, stream));
  }


  void sortAsync(const cudaStream_t stream){
    if(num_parts == 1){
      CHECK_CUDA(cudaMemcpyAsync(d_query_result_ids_sorted, d_query_result_ids,
                static_cast<size_t>(N_query) * K_query * sizeof(KeyT),
                cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_query_result_dists_sorted, d_query_result_dists,
            static_cast<size_t>(N_query) * K_query * sizeof(ValueT),
            cudaMemcpyDeviceToHost, stream));
      
    }
    else {
      cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_query_result_dists, d_query_result_dists_sorted, d_query_result_ids, d_query_result_ids_sorted,
      num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(ValueT)*8, stream);
    }
  }
};

#endif  // INCLUDE_GGNN_UTILS_CUDA_KNN_GGNN_QUERY_CUH_
