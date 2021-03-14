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

#ifndef INCLUDE_GGNN_CUDA_KNN_GGNN_GPU_INSTANCE_CUH_
#define INCLUDE_GGNN_CUDA_KNN_GGNN_GPU_INSTANCE_CUH_

#include <array>
#include <limits>
#include <string>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "ggnn/graph/cuda_knn_ggnn_graph_device.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph_host.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph_buffer.cuh"
#include "ggnn/merge/cuda_knn_merge_layer.cuh"
#include "ggnn/merge/cuda_knn_top_merge_layer.cuh"
#include "ggnn/query/cuda_knn_query_layer.cuh"
#include "ggnn/query/cuda_knn_ggnn_query.cuh"
#include "ggnn/query/cuda_knn_bf_query_layer.cuh"
#include "ggnn/query/cuda_knn_stats_query_layer.cuh"
#include "ggnn/select/cuda_knn_wrs_select_layer.cuh"
#include "ggnn/sym/cuda_knn_sym_buffer_merge_layer.cuh"
#include "ggnn/sym/cuda_knn_sym_query_layer.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"

template <typename ValueT>
__global__ void divide(ValueT* res, ValueT* input, ValueT N) {
  res[threadIdx.x] = input[threadIdx.x]/N;
}

/**
 * GGNN core operations (shared between single-GPU and multi-GPU version)
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
struct GGNNGPUInstance {
  /// number of base points per shard
  int N_shard;
  /// number of layers
  int L;
  /// growth factor (number of sub-graphs merged together per layer)
  int G;
  /// segment size in base layer
  int S0;
  /// number of segments in base layer with one additional element
  int S0_off;
  /// slack factor for symmetric linking
  float tau_build;

  /// total number of neighborhoods in the graph
  int N_all;
  /// total number of selection/translation entries
  int ST_all;

  /// neighborhoods per layer
  std::array<int, MAX_LAYER> Ns;  // [L]
  /// start of neighborhoods per layer
  std::array<int, MAX_LAYER> Ns_offsets;  // [L]
  /// start of selection/translation per layer
  std::array<int, MAX_LAYER> STs_offsets;  // [L]

  typedef GGNNGraphDevice<KeyT, BaseT, ValueT> GGNNGraphDevice;
  typedef GGNNGraphHost<KeyT, BaseT, ValueT> GGNNGraphHost;

  const Dataset<KeyT, BaseT, BAddrT>* dataset;
  GGNNGraphBuffer<KeyT, ValueT>* ggnn_buffer {nullptr};
  GGNNQuery<KeyT, ValueT, BaseT> ggnn_query;

  // Graph Shards resident on the GPU
  std::vector<GGNNGraphDevice> ggnn_shards;
  // Graph Shards resident on the CPU (for swapping, loading, and storing)
  std::vector<GGNNGraphHost> ggnn_cpu_buffers;

  curandGenerator_t gen;

  //TODO (lukas): merge the buffer-code in here?

  // CUDA GPU id associated with this instance
  const int gpu_id;

  // number of shards that need to be processed by this instance
  const int num_parts;

  GGNNGPUInstance(const int gpu_id, const Dataset<KeyT, BaseT, BAddrT>* dataset,
            const int N_shard, const int L,
            const bool enable_construction, const float tau_build,
            const int num_parts=1, const int num_cpu_buffers=1) :
    N_shard{N_shard}, L{L}, tau_build{tau_build},
    dataset{dataset}, gpu_id{gpu_id},
    ggnn_query{dataset->N_query, D, KQuery, num_parts},
    num_parts{num_parts}
  {
    CHECK_LE(L, MAX_LAYER);

    LOG(INFO) << "GGNNGPUInstance(): CUDA device id: " << gpu_id;
    {
      int current_gpu_id;
      cudaGetDevice(&current_gpu_id);
      CHECK_EQ(current_gpu_id, gpu_id) << "cudaSetDevice() needs to be called in advance!";
    }

    ggnn_query.loadQueriesAsync(dataset->h_query, 0);

    computeGraphParameters();

    CHECK_LE(static_cast<size_t>(N_all) * static_cast<size_t>(KBuild),
        static_cast<size_t>(std::numeric_limits<GAddrT>::max()))
      << "address type is insufficient to address the requested graph.";

    copyConstantsToGPU();

    // allocate CPU memory first (fail early if out of memory)
    ggnn_cpu_buffers.reserve(num_cpu_buffers);
    for (int i=0; i < num_cpu_buffers; i++)
      ggnn_cpu_buffers.emplace_back(N_shard, KBuild, N_all, ST_all);

    //TODO (lukas): merge the buffer-code in here?

    if (enable_construction)
      ggnn_buffer = new GGNNGraphBuffer<KeyT, ValueT>{N_shard, KBuild, KF};

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    int max_shards;
    {
      size_t free, total;
      CHECK_CUDA(cudaMemGetInfo(&free, &total));

      size_t size_per_shard = getSizePerShard();

      max_shards = free/size_per_shard;
      LOG(INFO) << "remaining device memory (" << free/(1024.0f*1024.0f*1024.0f)
                << " GB) suffices for " << max_shards << " shards ("
                << size_per_shard/(1024.0f*1024.0f*1024.0f) << " GB each).";

      CHECK_GT(max_shards, 0) << "use smaller shards.";
    }

    const int num_shards = min(max_shards, num_parts);
    ggnn_shards.reserve(num_shards);

    for (int i=0; i < num_shards; i++) {
      ggnn_shards.emplace_back(N_shard, D, KBuild, N_all, ST_all);
    }

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());
  }

  GGNNGPUInstance(const GGNNGPUInstance& other)
   : dataset{nullptr}, ggnn_query{0, D, KQuery},
     gpu_id{0}, N_shard{0}, num_parts{0} {
    // this exists to allow using vector::emplace_back
    // when it triggers a reallocation, this code will be called.
    // always make sure that enough memory is reserved ahead of time.
    LOG(FATAL) << "copying is not supported. reserve()!";
  }

   ~GGNNGPUInstance() {
     CHECK_CUDA(cudaSetDevice(gpu_id));
     ggnn_shards.clear();

     delete ggnn_buffer;

     CHECK_CUDA(cudaPeekAtLastError());
     CHECK_CUDA(cudaDeviceSynchronize());
     CHECK_CUDA(cudaPeekAtLastError());
   }

  void computeGraphParameters() {
    /// theoretical growth factor (number of sub-graphs merged together per
    /// layer)
    const float growth = powf(N_shard / static_cast<float>(S), 1.f / (L - 1));

    const int Gf = growth;
    const int Gc = growth + 1;

    const float S0f = N_shard / (pow(Gf, (L - 1)));
    const float S0c = N_shard / (pow(Gc, (L - 1)));

    const bool is_floor =
        (growth > 0) && ((S0c < KBuild) || (fabs(S0f - S) < fabs(S0c - S)));

    G = (is_floor) ? Gf : Gc;
    S0 = (is_floor) ? S0f : S0c;
    S0_off = N_shard - pow(G, L - 1) * S0;

    VLOG(1) << "GGNNGPUInstance(): N: " << N_shard << ", L: " << L
            << ", G: " << G << ", S: " << S << ", S0: " << S0
            << ", S0_off: " << S0_off << ", K: " << KBuild << ", KF: " << KF;

    N_all = 0;
    ST_all = 0;
    int N_current = N_shard;
    for (int l = 0; l < L; l++) {
      Ns[l] = N_current;
      Ns_offsets[l] = N_all;
      STs_offsets[l] = ST_all;
      N_all += N_current;
      if (l) {
        ST_all += N_current;
        N_current /= G;
      }
      else {
        N_current = S;
        for (int i=2;i<L; ++i)
          N_current *= G;
      }
    }
  }

  size_t getSizePerShard() const {
    const size_t graph_size = static_cast<GAddrT>(N_all) * KBuild * sizeof(KeyT);
    const size_t selection_translation_size = ST_all * sizeof(KeyT);
    // const size_t nn1_dist_buffer_size = N * sizeof(ValueT);
    const size_t nn1_stats_size = 2 * sizeof(ValueT);
    const size_t total_graph_size = graph_size + 2 * selection_translation_size
        + nn1_stats_size;
    const size_t base_size = static_cast<BAddrT>(N_shard) * D * sizeof(BaseT);

    return total_graph_size + base_size;
  }

  void copyConstantsToGPU() const {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    VLOG(2) << "GGNNGPUInstance::copyConstantsToGPU().\n";

    cudaMemcpyToSymbol(c_Ns, Ns.data(), L * sizeof(int));
    cudaMemcpyToSymbol(c_Ns_offsets, Ns_offsets.data(), L * sizeof(int));

    cudaMemcpyToSymbol(c_G, &G, sizeof(int));
    cudaMemcpyToSymbol(c_L, &L, sizeof(int));
    cudaMemcpyToSymbol(c_S0, &S0, sizeof(int));
    cudaMemcpyToSymbol(c_S0_offset, &S0_off, sizeof(int));

    cudaMemcpyToSymbol(c_tau_build, &tau_build, sizeof(float));
    cudaMemcpyToSymbol(c_STs_offsets, STs_offsets.data(), L * sizeof(int));
  }

  // graph utilities

  int getNs(const int layer) const { return Ns[layer]; }

  int getS(const int layer) const { return layer ? S : S0; }

  int getS_offset(const int layer) const { return layer ? 0 : S0_off; }

  KeyT* getGraph(const int shard, const int layer) {
    return &ggnn_shards.at(shard%ggnn_shards.size()).d_graph[static_cast<GAddrT>(Ns_offsets[layer]) * KBuild];
  }

  KeyT* getSelection(const int shard, const int layer) {
    if (!layer) {
      // there is no selection for layer 0
      return nullptr;
    }
    return &ggnn_shards.at(shard%ggnn_shards.size()).d_selection[STs_offsets[layer]];
  }

  KeyT* getTranslation(const int shard, const int layer) {
    if (!layer) {
      // there is no translation for layer 0
      return nullptr;
    }
    return &ggnn_shards.at(shard%ggnn_shards.size()).d_translation[STs_offsets[layer]];
  }

  // io

  void waitForDiskIO(const int shard_id) {
    auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];
    if (cpu_buffer.disk_io_thread.joinable())
      cpu_buffer.disk_io_thread.join();
  }

  void loadPartAsync(const std::string graph_dir, const int part_id, const int shard_id) {
    waitForDiskIO(shard_id);
    auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];
    auto load_part = [this, graph_dir, part_id, shard_id]() -> void {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());
      auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];

      cudaStreamSynchronize(shard.stream);

      if (shard.current_part_id == part_id) {
        VLOG(4) << "[GPU: " << gpu_id << "] part " << part_id << " is already loaded on shard " << shard_id;
        return;
      }

      shard.current_part_id = part_id;

      loadShardBaseDataAsync(part_id, shard_id);

      if (cpu_buffer.current_part_id == part_id) {
        VLOG(4) << "[GPU: " << gpu_id << "] part " << part_id << " is already loaded on cpu buffer " << shard_id%ggnn_cpu_buffers.size();
      }
      else {
        const std::string part_filename = graph_dir + "part_" + std::to_string(part_id) + ".ggnn";
        cpu_buffer.load(part_filename);
        VLOG(2) << "[GPU: " << gpu_id << "] loaded part " << part_id << " from " << part_filename.c_str();
        cpu_buffer.current_part_id = part_id;
      }

      cpu_buffer.uploadAsync(shard);
      cudaStreamSynchronize(shard.stream);
      VLOG(4) << "[GPU: " << gpu_id << "] uploaded part " << part_id;
    };
    cpu_buffer.disk_io_thread = std::thread(load_part);
  }

  void uploadPartAsync(const int part_id, const int shard_id) {
    waitForDiskIO(shard_id);
    auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];
    auto upload_part = [this, part_id, shard_id]() -> void {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());
      auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];

      cudaStreamSynchronize(shard.stream);

      if (shard.current_part_id == part_id) {
        VLOG(4) << "[GPU: " << gpu_id << "] part " << part_id << " is already loaded on shard " << shard_id;
        return;
      }

      shard.current_part_id = part_id;
      CHECK_EQ(cpu_buffer.current_part_id, part_id);

      loadShardBaseDataAsync(part_id, shard_id);
      cpu_buffer.uploadAsync(shard);
      cudaStreamSynchronize(shard.stream);
      VLOG(4) << "[GPU: " << gpu_id << "] uploaded part " << part_id;
    };
    cpu_buffer.disk_io_thread = std::thread(upload_part);
  }

  void storePartAsync(const std::string graph_dir, const int part_id, const int shard_id) {
    waitForDiskIO(shard_id);
    auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];
    auto store_part = [this, graph_dir, part_id, shard_id]() -> void {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());
      auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];

      if (cpu_buffer.current_part_id == part_id) {
        VLOG(4) << "[GPU: " << gpu_id << "] part " << part_id << " is already downloaded";
      }
      else {
        cpu_buffer.downloadAsync(shard);
        cudaStreamSynchronize(shard.stream);
        VLOG(4) << "[GPU: " << gpu_id << "] downloaded part " << part_id;
      }

      const std::string part_filename = graph_dir + "part_" + std::to_string(part_id) + ".ggnn";
      cpu_buffer.store(part_filename);
      VLOG(2) << "[GPU: " << gpu_id << "] stored part " << part_id << " to " << part_filename.c_str();
    };
    cpu_buffer.disk_io_thread = std::thread(store_part);
  }

  void downloadPartAsync(const int part_id, const int shard_id) {
    waitForDiskIO(shard_id);
    auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];
    auto download_part = [this, part_id, shard_id]() -> void {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());
      auto& cpu_buffer = ggnn_cpu_buffers[shard_id%ggnn_cpu_buffers.size()];

      cpu_buffer.downloadAsync(shard);
      cudaStreamSynchronize(shard.stream);
      cpu_buffer.current_part_id = part_id;
      VLOG(4) << "[GPU: " << gpu_id << "] downloaded part " << part_id;
    };
    cpu_buffer.disk_io_thread = std::thread(download_part);
  }

  void loadShardBaseDataAsync(const int part_id, const int shard_id) {
    const size_t memsize = static_cast<BAddrT>(N_shard) * D * sizeof(BaseT);
    const size_t N_offset = static_cast<BAddrT>(N_shard) * part_id;
    auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());
    CHECK_CUDA(cudaMemcpyAsync(shard.d_base, dataset->h_base + N_offset * D,
                               memsize, cudaMemcpyHostToDevice, shard.stream));
  }

  void generateGTUsingBF(const int shard_id = 0) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    const int KGT = 100;
    KeyT* m_gt = nullptr;
    CHECK_CUDA(cudaMallocManaged(&m_gt, sizeof(KeyT)*KGT*dataset->N_query));

    CHECK_LE(dataset->K_gt, KGT) << "The brute force query is set to " << KGT << " neighbors, but the dataset is configured for " << dataset->K_gt << ".";

    typedef BruteForceQueryKernel<measure, ValueT, KeyT, D, KGT, 32,
                                  BaseT, BAddrT, GAddrT, false>
        QueryKernel;

    LOG(INFO) << "Running brute force query to determine ground truth";

    QueryKernel query_kernel;
    query_kernel.d_base = shard.d_base;
    query_kernel.d_query = ggnn_query.d_query;

    query_kernel.d_query_results = m_gt;

    query_kernel.N_base = N_shard; // this applies to potential subsets
    query_kernel.N = dataset->N_query;
    query_kernel.N_offset = 0;

    time_launcher(0, &query_kernel, query_kernel.N, shard.stream);

    cudaStreamSynchronize(shard.stream);

    if (dataset->K_gt == KGT) {
      std::copy_n(m_gt, KGT*dataset->N_query, dataset->gt);
    }
    else {
      const size_t stride_results = static_cast<size_t>(dataset->N_query)*KGT;
      const size_t stride_dest = static_cast<size_t>(dataset->N_query)*dataset->K_gt;
      for (int n=0; n<dataset->N_query; ++n) {
        std::copy_n(m_gt+n*stride_results, dataset->K_gt, dataset->gt+n*stride_dest);
      }
    }

    CHECK_CUDA(cudaFree(m_gt));

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());
  }

  // graph operations

  template <int BLOCK_DIM_X = 32, int MAX_ITERATIONS = 400, int CACHE_SIZE = 512, int SORTED_SIZE = 256, bool DIST_STATS = false>
  void queryLayer(const int shard_id = 0) const {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    typedef QueryKernel<measure, ValueT, KeyT, D, KBuild, KF, KQuery, S, BLOCK_DIM_X, BaseT,
                        BAddrT, GAddrT, DIST_STATS, false, MAX_ITERATIONS, CACHE_SIZE, SORTED_SIZE, true>
        QueryKernel;

    int* m_dist_statistics = nullptr;
    if (DIST_STATS)
      cudaMallocManaged(&m_dist_statistics, dataset->N_query * sizeof(int));

    QueryKernel query_kernel;
    query_kernel.d_base = shard.d_base;
    query_kernel.d_query = ggnn_query.d_query;

    query_kernel.d_graph = shard.d_graph;
    query_kernel.d_query_results = ggnn_query.d_query_result_ids;
    query_kernel.d_query_results_dists = ggnn_query.d_query_result_dists;

    query_kernel.d_translation = shard.d_translation;

    query_kernel.d_nn1_stats = shard.d_nn1_stats;

    query_kernel.N = dataset->N_query;
    query_kernel.N_offset = 0;

    query_kernel.d_dist_stats = m_dist_statistics;

    query_kernel.part = shard_id;
    query_kernel.num_parts = num_parts;
    query_kernel.N_base = N_shard;

    query_kernel.launch(shard.stream);

    if (DIST_STATS)
      cudaFree(m_dist_statistics);
  }

  void select(const int layer, const int shard_id = 0) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    typedef WRSSelectionKernel<ValueT, KeyT, 128, S> SelectionKernel;

    SelectionKernel select_kernel;

    select_kernel.d_selection = getSelection(shard_id, layer + 1);
    select_kernel.d_translation = getTranslation(shard_id, layer + 1);

    select_kernel.d_translation_layer = getTranslation(shard_id, layer);

    select_kernel.layer = layer;

    select_kernel.S = getS(layer);
    select_kernel.S_offset = getS_offset(layer);

    const int SG = S / G;
    const int SG_offset = S - SG * G;

    select_kernel.SG = SG;
    select_kernel.SG_offset = SG_offset;

    select_kernel.B = pow(G, L - 1 - layer);
    select_kernel.B_offset = 0;

    select_kernel.d_rng = ggnn_buffer->d_rng;
    select_kernel.d_nn1_dist_buffer = ggnn_buffer->d_nn1_dist_buffer;

    /* Generate n floats on device */
    curandGenerateUniform(gen, ggnn_buffer->d_rng, getNs(layer));

    time_launcher(2, &select_kernel, getNs(layer), shard.stream);
  }

  void top(const int layer, const int shard_id = 0) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    typedef TopMergeKernel<measure, ValueT, KeyT, D, KBuild, 128, BaseT, BAddrT, GAddrT>
        TopMergeKernel;

    TopMergeKernel top_kernel;
    top_kernel.d_base = shard.d_base;
    top_kernel.d_translation = getTranslation(shard_id, layer);
    top_kernel.d_graph = getGraph(shard_id, layer);
    top_kernel.d_nn1_dist_buffer = ggnn_buffer->d_nn1_dist_buffer;

    top_kernel.layer = layer;

    top_kernel.N = getNs(layer);
    top_kernel.N_offset = 0;

    top_kernel.S = getS(layer);
    top_kernel.S_offset = getS_offset(layer);

    time_launcher(2, &top_kernel, getNs(layer), shard.stream);
  }

  void mergeLayer(const int layer_top, const int layer_btm, const int shard_id = 0) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    typedef MergeKernel<measure, ValueT, KeyT, D, KBuild, KF, S, 32, BaseT, BAddrT,
                        GAddrT>
        MergeKernel;

    const size_t graph_buffer_size =
        static_cast<GAddrT>(getNs(layer_btm)) * KBuild *
        sizeof(KeyT);

    MergeKernel merge_kernel;
    merge_kernel.d_base = shard.d_base;

    merge_kernel.d_graph = shard.d_graph;
    merge_kernel.d_graph_buffer = ggnn_buffer->d_graph_buffer;

    merge_kernel.d_translation = shard.d_translation;
    merge_kernel.d_selection = shard.d_selection;

    merge_kernel.d_nn1_stats = shard.d_nn1_stats;
    merge_kernel.d_nn1_dist_buffer = ggnn_buffer->d_nn1_dist_buffer;

    merge_kernel.N = getNs(layer_btm);
    merge_kernel.N_offset = 0;

    merge_kernel.layer_top = layer_top;
    merge_kernel.layer_btm = layer_btm;

    time_launcher(2, &merge_kernel, getNs(layer_btm), shard.stream);

    cudaMemcpyAsync((void*)getGraph(shard_id, layer_btm), (void*)ggnn_buffer->d_graph_buffer,
               graph_buffer_size, cudaMemcpyDeviceToDevice, shard.stream);
  };

  void merge(const int layer_top, const int layer_btm, const int shard_id = 0) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    VLOG(2) << "merge: " << layer_top << layer_btm << std::endl;
    if (layer_top == layer_btm)
      top(layer_btm, shard_id);
    else
      mergeLayer(layer_top, layer_btm, shard_id);

    if (!layer_btm)
      computeNN1Stats(shard_id);
  };

  void computeNN1Stats(const int shard_id = 0) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    CHECK_CUDA(cub::DeviceReduce::Sum(ggnn_buffer->d_temp_storage_sum,
                                      ggnn_buffer->temp_storage_bytes_sum,
                                      ggnn_buffer->d_nn1_dist_buffer,
                                      &shard.d_nn1_stats[0], N_shard,
                                      shard.stream));

    divide<ValueT><<<1, 1, 0, shard.stream>>>(shard.d_nn1_stats,
        shard.d_nn1_stats, ValueT(N_shard));

    CHECK_CUDA(cub::DeviceReduce::Max(ggnn_buffer->d_temp_storage_max,
                                      ggnn_buffer->temp_storage_bytes_max,
                                      ggnn_buffer->d_nn1_dist_buffer,
                                      &shard.d_nn1_stats[1], N_shard,
                                      shard.stream));

    if(VLOG_IS_ON(2))
    {
      ValueT h_nn1_stats[2];
      cudaMemcpyAsync(h_nn1_stats, shard.d_nn1_stats, 2*sizeof(ValueT), cudaMemcpyDeviceToHost, shard.stream);
      cudaStreamSynchronize(shard.stream);
      VLOG(2) << "mean: " << h_nn1_stats[0] << " | max: " << h_nn1_stats[1] << std::endl;
    }
  }

  void sym(const int layer, const int shard_id = 0) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    const auto& shard = ggnn_shards.at(shard_id%ggnn_shards.size());

    typedef SymQueryKernel<measure, ValueT, KeyT, D, KBuild, KF, 64, BaseT, BAddrT,
                           GAddrT>
        SymQueryKernel;

    cudaMemsetAsync(
        ggnn_buffer->d_sym_buffer, -1,
        static_cast<GAddrT>(static_cast<GAddrT>(getNs(layer))) *
            KF * sizeof(KeyT), shard.stream);

    cudaMemsetAsync(ggnn_buffer->d_sym_atomic, 0, getNs(layer) * sizeof(int), shard.stream);

    SymQueryKernel sym_kernel;

    sym_kernel.d_base = shard.d_base;
    sym_kernel.d_graph = getGraph(shard_id, layer);
    sym_kernel.d_translation = getTranslation(shard_id, layer);

    sym_kernel.d_sym_atomic = ggnn_buffer->d_sym_atomic;
    sym_kernel.d_sym_buffer = ggnn_buffer->d_sym_buffer;

    sym_kernel.d_nn1_stats = shard.d_nn1_stats;
    sym_kernel.d_stats = ggnn_buffer->d_statistics;

    sym_kernel.layer = layer;

    sym_kernel.N = getNs(layer);

    sym_kernel.N_offset = 0;

    // CHECK_CUDA(cudaPeekAtLastError());
    // CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaPeekAtLastError());

    time_launcher(2, &sym_kernel, getNs(layer), shard.stream);

    // CHECK_CUDA(cudaPeekAtLastError());
    // CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaPeekAtLastError());

    typedef SymBufferMergeKernel<ValueT, KeyT, KBuild, KF, 128, GAddrT>
        SymBufferMergeKernel;
    SymBufferMergeKernel sym_buffer_merge_kernel;

    sym_buffer_merge_kernel.d_sym_buffer = ggnn_buffer->d_sym_buffer;
    sym_buffer_merge_kernel.d_sym_atomic = ggnn_buffer->d_sym_atomic;
    sym_buffer_merge_kernel.d_graph = getGraph(shard_id, layer);

    sym_buffer_merge_kernel.N = getNs(layer);
    sym_buffer_merge_kernel.N_offset = 0;

    time_launcher(3, &sym_buffer_merge_kernel, getNs(layer), shard.stream);

    // CHECK_CUDA(cudaPeekAtLastError());
    // CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaPeekAtLastError());

    if(VLOG_IS_ON(2)){
      int* h_sym_atomic;
      //int* h_statistics;

      CHECK_CUDA(cudaMallocHost(&h_sym_atomic, static_cast<size_t>(getNs(layer)) * sizeof(int)));
      //CHECK_CUDA(cudaMallocHost(&h_statistics, static_cast<size_t>(getNs(layer)) * sizeof(int)));

      cudaMemcpyAsync(h_sym_atomic, ggnn_buffer->d_sym_atomic, static_cast<size_t>(getNs(layer)) * sizeof(int), cudaMemcpyDeviceToHost, shard.stream);
      //cudaMemcpyAsync(h_statistics, ggnn_buffer->d_statistics, static_cast<size_t>(getNs(layer)) * sizeof(int), cudaMemcpyDeviceToHost, shard.stream);

      cudaStreamSynchronize(shard.stream);

      int c = 0;
      int m = 0;
      // int unconnected = 0;
      for (int i = 0; i < getNs(layer); i++) {
        if (h_sym_atomic[i] > KF) c++;
        m += (h_sym_atomic[i] > KF) ? KF : h_sym_atomic[i];
        // unconnected += h_statistics[i];
      }
      VLOG(2) << "Layer " << layer
              << " [N: " << getNs(layer)
              << "] | overflow: " << c << " (" << c / float(getNs(layer))
              << ") | added_links: " << m << " (" << m / float(getNs(layer))
              << ") || unconnected: OVERFLOW_STATS currently not computed. )\n";

      cudaFreeHost(h_sym_atomic);
    }

    // cudaFree(d_sym_buffer);
    // cudaFree(m_sym_atomic);
    // cudaFree(m_statistics);

    // CHECK_CUDA(cudaPeekAtLastError());
    // CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaPeekAtLastError());
  };

  void build(const int part_id, const int shard_id = 0) {
    CHECK(ggnn_buffer) << "the construction buffer is not allocated.";

    VLOG(1) << "build(): part_id: " << part_id << " shard_id: " << shard_id;
    for (int layer_top = 0; layer_top < L; layer_top++) {
      for (int layer_btm = layer_top; layer_btm >= 0; layer_btm--) {
        VLOG(2) << "layer_top: " << layer_top << " -> layer_btm: " << layer_btm << std::endl;

        merge(layer_top, layer_btm, shard_id);

        if (layer_top < (L - 1) && layer_top == layer_btm)
          select(layer_top, shard_id);

        sym(layer_btm, shard_id);
      }
    }
  }

  void refine(const int shard_id = 0) {
    for (int layer = L - 2; layer >= 0; layer--) {
      merge(L - 1, layer, shard_id);
      sym(layer, shard_id);
    }
  }
};

#endif  // INCLUDE_GGNN_CUDA_KNN_GGNN_GPU_INSTANCE_CUH_
