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

#ifndef INCLUDE_GGNN_CUDA_KNN_GGNN_CUH_
#define INCLUDE_GGNN_CUDA_KNN_GGNN_CUH_

#include <limits>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cub/cub.cuh"
#include "ggnn/cuda_knn_ggnn_gpu_instance.cuh"
#include "ggnn/query/cuda_knn_query_layer.cuh"
#include "ggnn/query/cuda_knn_bf_query_layer.cuh"
#include "ggnn/query/cuda_knn_stats_query_layer.cuh"
#include "ggnn/query/cuda_knn_no_slack_query_layer.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"
#include "ggnn/utils/cuda_knn_ggnn_results.cuh"


// for storing generated ground truth data
#include "io/storer_ann.hpp"

// only needed for file_exists check
#include <sys/stat.h>

inline bool file_exists(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

/**
 * GGNN single-GPU wrapper
 *
 * @param measure distance measure: Euclidean or Cosine
 * @param KeyT datatype of dataset indices (needs to be able to represent
 * N_base, signed integer required)
 * @param ValueT distance value type
 * @param GAddrT address type used to access neighborhood vectors (needs to be
 * able to represent N_all*K)
 * @param BaseT datatype of dataset vector elements
 * @param BAddrT address type used to access dataset vectors (needs to be able
 * to represent N_base*D)
 * @param D dimension of dataset
 * @param KBuild neighbors per node in the GGNN graph
 * @param KF maximum number of inverse links per node in the GGNN graph
 * @param KQuery number of nearest neighbors to retrieve during query
 * @param S segment size
 */
template <DistanceMeasure measure,
          typename KeyT, typename ValueT, typename GAddrT, typename BaseT,
          typename BAddrT, int D, int KBuild, int KF, int KQuery, int S>
struct GGNN {
  using Dataset = Dataset<KeyT, BaseT, BAddrT>;
  using GGNNGPUInstance = GGNNGPUInstance<measure, KeyT, ValueT, GAddrT, BaseT, BAddrT, D, KBuild, KF, KQuery, S>;
  using GGNNResults = GGNNResults<measure, KeyT, ValueT, BaseT, BAddrT, KQuery>;

  Dataset dataset;
  GGNNGPUInstance ggnn_gpu_instance;
  GGNNResults ggnn_results {&dataset};

  GGNN(const std::string& basePath, const std::string& queryPath,
       const std::string& gtPath, const int L, const float tau_build,
       const size_t N_base = std::numeric_limits<size_t>::max())
      : dataset{basePath, queryPath, file_exists(gtPath) ? gtPath : "", N_base},
        ggnn_gpu_instance{[](){int device; cudaGetDevice(&device); return device;}(), &dataset, dataset.N_base, L, true, tau_build} {
    CHECK_EQ(dataset.D, D) << "DIM needs to be the same";

    const auto& shard = ggnn_gpu_instance.ggnn_shards.at(0);
    ggnn_gpu_instance.loadShardBaseDataAsync(0, 0);
    cudaStreamSynchronize(shard.stream);

    if (gtPath.empty() || !file_exists(gtPath)) {
      generateGTUsingBF();
      if (!gtPath.empty()) {
        LOG(INFO) << "exporting brute-forced ground truth data.";
        IVecsStorer gt_storer(gtPath, dataset.K_gt,
            dataset.N_query);
        gt_storer.store(dataset.gt, dataset.N_query);
      }
    }
  }

  void ggnnMain(const std::string& graph_filename, const int refinement_iterations) {
    const bool export_graph =
        !graph_filename.empty() && !file_exists(graph_filename);
    const bool import_graph =
        !graph_filename.empty() && file_exists(graph_filename);
    const bool perform_build = export_graph || !import_graph;

    if (perform_build) {
      std::vector<float> construction_times;
      construction_times.reserve(refinement_iterations+1);

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      LOG(INFO) << "Starting Graph construction... (tau=" << ggnn_gpu_instance.tau_build << ")";

      cudaEventRecord(start);
      build();
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      construction_times.push_back(milliseconds);

      for (int refinement_step = 0; refinement_step < refinement_iterations;
           ++refinement_step) {
        DLOG(INFO) << "Refinement step " << refinement_step;
        refine();
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float elapsed_milliseconds = 0;
        cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
        construction_times.push_back(elapsed_milliseconds);
      }
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      for (int refinement_step = 0;
           refinement_step < construction_times.size(); refinement_step++) {
        const float elapsed_milliseconds = construction_times[refinement_step];
        const float elapsed_seconds = elapsed_milliseconds / 1000.0f;
        const int number_of_points = ggnn_gpu_instance.N_shard;

        LOG(INFO) << "Graph construction + " << refinement_step << " refinement step(s)";
        LOG(INFO) << "                   -- secs: " << elapsed_seconds;
        LOG(INFO) << "                   -- points: " << number_of_points;
        LOG(INFO) << "                   -- ms/point: "
                  << elapsed_milliseconds / number_of_points;
      }

      if (export_graph) {
        write(graph_filename);
      }
    }

    if (import_graph) {
      read(graph_filename);
    }
  }

  /**
   * reset the graph and prepare for a subset of size N
   */
  void reinit_graph_for_subset(KeyT N) {
    CHECK_LE(N, dataset.N_base);
    ggnn_gpu_instance.N_shard = N;
    ggnn_gpu_instance.computeGraphParameters();
    ggnn_gpu_instance.copyConstantsToGPU();

    dataset.top1DuplicateEnd.clear();
    dataset.topKDuplicateEnd.clear();
  }

  void read(const std::string& filename) {
    auto& ggnn_host = ggnn_gpu_instance.ggnn_cpu_buffers.at(0);
    auto& ggnn_device = ggnn_gpu_instance.ggnn_shards.at(0);

    ggnn_host.load(filename);

    ggnn_host.uploadAsync(ggnn_device);
    cudaStreamSynchronize(ggnn_device.stream);
  }

  void write(const std::string& filename) {
    auto& ggnn_host = ggnn_gpu_instance.ggnn_cpu_buffers.at(0);
    auto& ggnn_device = ggnn_gpu_instance.ggnn_shards.at(0);

    ggnn_host.downloadAsync(ggnn_device);
    cudaStreamSynchronize(ggnn_device.stream);

    ggnn_host.store(filename);
  }

  void evaluateKNNGraph() {
    CHECK_EQ(dataset.N_base, dataset.N_query) << "the base needs to be loaded as the query set.";
    CHECK_GE(KBuild/2, KQuery) << "there aren't as many nearest neighbors in the graph as queried for.";
    CHECK_GE(dataset.K_gt, KQuery+1) << "need one additional ground truth entry to exclude the point itself.";

    KeyT* const original_gt = dataset.gt;
    dataset.top1DuplicateEnd.clear();
    dataset.topKDuplicateEnd.clear();
    dataset.gt = new KeyT[static_cast<size_t>(dataset.N_query)*dataset.K_gt];

    // shift ground truth left by one to exclude the point itself
    std::copy_n(original_gt+1, static_cast<size_t>(dataset.N_query)*dataset.K_gt-1, dataset.gt);

    dataset.template checkForDuplicatesInGroundTruth<measure, ValueT>(KQuery);

    auto& ggnn_host = ggnn_gpu_instance.ggnn_cpu_buffers.at(0);
    auto& ggnn_device = ggnn_gpu_instance.ggnn_shards.at(0);

    ggnn_host.downloadAsync(ggnn_device);
    cudaStreamSynchronize(ggnn_device.stream);

    // simply copy the neighbors from the graph into the results
    for (size_t n=0; n<dataset.N_query; ++n) {
      std::copy_n(ggnn_host.h_graph+n*KBuild, KQuery, ggnn_results.h_sorted_ids+n*KQuery);
    }

    ggnn_results.evaluateResults();

    delete[] dataset.gt;
    dataset.gt = original_gt;
  }

  template <int BLOCK_DIM_X = 32, int MAX_ITERATIONS = 400, int CACHE_SIZE = 512, int SORTED_SIZE = 256, bool DIST_STATS = false>
  void queryLayer() {
    dataset.template checkForDuplicatesInGroundTruth<measure, ValueT>(KQuery);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    const auto& shard = ggnn_gpu_instance.ggnn_shards.at(0);

    cudaEventRecord(start, shard.stream);
    ggnn_gpu_instance.template queryLayer<BLOCK_DIM_X, MAX_ITERATIONS, CACHE_SIZE, SORTED_SIZE, DIST_STATS>();
    cudaEventRecord(stop, shard.stream);
    ggnn_gpu_instance.ggnn_query.sortAsync(shard.stream);
    ggnn_results.loadAsync(ggnn_gpu_instance.ggnn_query, 0, shard.stream);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    VLOG(0) << "[GPU: " << ggnn_gpu_instance.gpu_id << "] query part: " << 0 << " => ms: " << milliseconds << " [" << dataset.N_query << " points query -> " << milliseconds*1000.0f/dataset.N_query << " us/point] \n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamSynchronize(shard.stream);
    ggnn_results.merge();
    ggnn_results.evaluateResults();
  }

  template <int BLOCK_DIM_X = 32, int MAX_ITERATIONS = 400, int CACHE_SIZE = 512, int SORTED_SIZE = 256, int BEST_SIZE = 128, bool DIST_STATS = false>
  void noSlackQueryLayer() {
    dataset.template checkForDuplicatesInGroundTruth<measure, ValueT>(KQuery);

    auto& shard = ggnn_gpu_instance.ggnn_shards.at(0);

    typedef NoSlackQueryKernel<measure, ValueT, KeyT, D, KBuild, KF, KQuery, S, BLOCK_DIM_X, BaseT,
                        BAddrT, GAddrT, DIST_STATS, false, MAX_ITERATIONS, CACHE_SIZE, SORTED_SIZE, BEST_SIZE>
        QueryKernel;

    KeyT* m_query_results;
    cudaMallocManaged(&m_query_results,
                      dataset.N_query * KQuery * sizeof(KeyT));
    int* m_dist_statistics = nullptr;
    if (DIST_STATS)
      cudaMallocManaged(&m_dist_statistics, dataset.N_query * sizeof(int));

    QueryKernel query_kernel;
    query_kernel.d_base = shard.d_base;
    query_kernel.d_query = ggnn_gpu_instance.ggnn_query.d_query;

    query_kernel.d_graph = shard.d_graph;
    query_kernel.d_query_results = ggnn_gpu_instance.ggnn_query.d_query_result_ids;

    query_kernel.d_translation = shard.d_translation;

    query_kernel.d_nn1_stats = shard.d_nn1_stats;

    query_kernel.N = dataset.N_query;
    query_kernel.N_offset = 0;

    query_kernel.d_dist_stats = m_dist_statistics;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start, shard.stream);
    query_kernel.launch(shard.stream);
    cudaEventRecord(stop, shard.stream);
    ggnn_gpu_instance.ggnn_query.sortAsync(shard.stream);
    ggnn_results.loadAsync(ggnn_gpu_instance.ggnn_query, 0, shard.stream);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    VLOG(0) << "[GPU: " << ggnn_gpu_instance.gpu_id << "] query part: " << 0 << " => ms: " << milliseconds << " [" << dataset.N_query << " points query -> " << milliseconds*1000.0f/dataset.N_query << " us/point] \n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamSynchronize(shard.stream);
    ggnn_results.merge();
    ggnn_results.evaluateResults();

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());
  }

  /// verbose query with additional logging
  /// templated mainly to avoid compilation when not used
  template <int BLOCK_DIM_X = 32, int MAX_ITERATIONS = 400, int CACHE_SIZE = 512, int SORTED_SIZE = 256>
  void queryLayerDebug() {
    dataset.template checkForDuplicatesInGroundTruth<measure, ValueT>(KQuery);

    auto& shard = ggnn_gpu_instance.ggnn_shards.at(0);

    /*
    typedef QueryKernel<ValueT, KeyT, D, KBuild, KF, KQuery, S, BLOCK_DIM_X,
                                BaseT, BAddrT, GAddrT, true, false, MAX_ITERATIONS, CACHE_SIZE, SORTED_SIZE, true>
    */
    typedef StatsQueryKernel<measure, ValueT, KeyT, D, KBuild, KF, KQuery, S, BLOCK_DIM_X, BaseT,
                             BAddrT, GAddrT, true, false, MAX_ITERATIONS, CACHE_SIZE, SORTED_SIZE>
        QueryKernel;


    KeyT* m_query_results;
    cudaMallocManaged(&m_query_results,
                      dataset.N_query * KQuery * sizeof(KeyT));
    ValueT* m_query_results_dists;
    cudaMallocManaged(&m_query_results_dists,
                      dataset.N_query * KQuery * sizeof(ValueT));
    int* m_dist_statistics;
    cudaMallocManaged(&m_dist_statistics, dataset.N_query * sizeof(int));

    ValueT* m_dist_1_best_stats;
    ValueT* m_dist_k_best_stats;
    cudaMallocManaged(&m_dist_1_best_stats,
                      dataset.N_query * (MAX_ITERATIONS+1) * sizeof(ValueT));
    cudaMallocManaged(&m_dist_k_best_stats,
                      dataset.N_query * (MAX_ITERATIONS+1) * sizeof(ValueT));
    cudaMemset(m_dist_1_best_stats, -1, dataset.N_query * (MAX_ITERATIONS+1) * sizeof(ValueT));
    cudaMemset(m_dist_k_best_stats, -1, dataset.N_query * (MAX_ITERATIONS+1) * sizeof(ValueT));

    const KeyT debug_query_id = -1;
    KeyT* m_debug_query_visited_ids;
    if (debug_query_id > 0) {
      cudaMallocManaged(&m_debug_query_visited_ids, MAX_ITERATIONS * sizeof(KeyT));
      cudaMemset(m_debug_query_visited_ids, -1, MAX_ITERATIONS * sizeof(KeyT));
    }

    QueryKernel query_kernel;
    query_kernel.d_base = shard.d_base;
    query_kernel.d_query = ggnn_gpu_instance.ggnn_query.d_query;

    query_kernel.d_graph = shard.d_graph;
    query_kernel.d_query_results = m_query_results;
    query_kernel.d_query_results_dists = m_query_results_dists;

    query_kernel.d_dist_1_best_stats = m_dist_1_best_stats;
    query_kernel.d_dist_k_best_stats = m_dist_k_best_stats;
    query_kernel.d_debug_query_visited_ids = m_debug_query_visited_ids;
    query_kernel.debug_query_id = debug_query_id;

    query_kernel.d_translation = shard.d_translation;

    query_kernel.d_nn1_stats = shard.d_nn1_stats;

    //query_kernel.N_base = dataset.N_base;
    query_kernel.N = dataset.N_query;
    query_kernel.N_offset = 0;

    query_kernel.d_dist_stats = m_dist_statistics;

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());

    time_launcher(0, &query_kernel, query_kernel.N);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());

    std::ofstream distance_stats_file("distances_k_best.csv", std::ofstream::out);
    distance_stats_file << "top-layer;";
    for (int j=0; j<MAX_ITERATIONS; ++j)
      distance_stats_file << "iteration " << j << ";";
    distance_stats_file << "last improvement;last distance" << std::endl;
    for (int i=0; i<dataset.N_query; ++i) {
      ValueT last_dist = std::numeric_limits<ValueT>::infinity();
      int last_improvement = 0;
      for (int j=0; j<MAX_ITERATIONS+1; ++j) {
        const ValueT dist = m_dist_k_best_stats[i*(MAX_ITERATIONS+1)+j];
        distance_stats_file << dist << ";";
        if (dist < last_dist) {
          last_dist = dist;
          last_improvement = j;
        }
      }
      distance_stats_file << last_improvement << ";" << last_dist << std::endl;
    }
    distance_stats_file.close();

    if (debug_query_id > 0) {
      // compute distance matrix for multi dimensional scaling
      std::vector<ValueT> distance_matrix;
      // wasteful, but easier than indexing a triangle matrix
      distance_matrix.resize(MAX_ITERATIONS*MAX_ITERATIONS, std::numeric_limits<ValueT>::infinity());
      for (int i=0; i<MAX_ITERATIONS; ++i) {
        for (int j=i+1; j<MAX_ITERATIONS; ++j) { // this will take some time
          distance_matrix[i*MAX_ITERATIONS+j] = dataset.template compute_distance_base_to_base<measure, ValueT>(m_debug_query_visited_ids[i], m_debug_query_visited_ids[j]);
        }
      }

      std::vector<ValueT> distances_to_query;
      distances_to_query.resize(MAX_ITERATIONS);
      std::ofstream visited_distance_matrix_file("visited_distance_matrix.csv", std::ofstream::out);
      visited_distance_matrix_file << ValueT(0);
      for (int i=0; i<MAX_ITERATIONS; ++i) {
        distances_to_query[i] = dataset.template compute_distance_query<measure, ValueT>(m_debug_query_visited_ids[i], query_kernel.debug_query_id);
        visited_distance_matrix_file << ';' << distances_to_query[i];
      }
      visited_distance_matrix_file << std::endl;
      for (int i=0; i<MAX_ITERATIONS; ++i) {
        // insert query point as first point
        visited_distance_matrix_file << distances_to_query[i];

        for (int j=0; j<MAX_ITERATIONS; ++j) {
          visited_distance_matrix_file << ';';
          if (j<i)
            visited_distance_matrix_file << distance_matrix[j*MAX_ITERATIONS+i];
          else if (i < j)
            visited_distance_matrix_file << distance_matrix[i*MAX_ITERATIONS+j];
          else // if (i == j)
            visited_distance_matrix_file << 0;
        }
        visited_distance_matrix_file << std::endl;
      }
      visited_distance_matrix_file.close();
    }

    printf("query results:\n");
    for (int i=0; i<min(100, dataset.N_query); ++i) {
      KeyT gt_index = dataset.gt[i*dataset.K_gt];
      printf("query %i:", i);
      for (int j=0; j<KQuery; ++j) {
        ValueT result_distance = measure == Euclidean ? sqrtf(m_query_results_dists[i*KQuery+j]) : m_query_results_dists[i*KQuery+j];
        printf("\t%i (%f)", m_query_results[i*KQuery+j], result_distance);
      }
      printf("\tgt: %i (%f)\n", gt_index, dataset.template compute_distance_query<measure, ValueT>(gt_index, i));
    }

    std::copy_n(m_query_results, static_cast<size_t>(dataset.N_query)*KQuery, ggnn_results.h_sorted_ids);
    ggnn_results.evaluateResults();

    cudaFree(m_query_results);
    cudaFree(m_query_results_dists);
    cudaFree(m_dist_statistics);
    cudaFree(m_dist_1_best_stats);
    cudaFree(m_dist_k_best_stats);
    if (debug_query_id > 0)
      cudaFree(m_debug_query_visited_ids);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());
  }

  void generateGTUsingBF() {
    ggnn_gpu_instance.generateGTUsingBF(0);
  }

  void build() {
    ggnn_gpu_instance.build(0);
  }

  void refine() {
    ggnn_gpu_instance.refine();
  }
};

#endif  // INCLUDE_GGNN_CUDA_KNN_GGNN_CUH_
