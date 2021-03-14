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

#ifndef INCLUDE_GGNN_CUDA_KNN_GGNN_MULTI_GPU_CUH_
#define INCLUDE_GGNN_CUDA_KNN_GGNN_MULTI_GPU_CUH_

#include <chrono>
#include <limits>
#include <string>
#include <thread>
#include <stdio.h>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cub/cub.cuh"
#include "ggnn/cuda_knn_ggnn_gpu_instance.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph_device.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph_host.cuh"
#include "ggnn/query/cuda_knn_query_layer.cuh"
#include "ggnn/query/cuda_knn_ggnn_query.cuh"
#include "ggnn/query/cuda_knn_bf_query_layer.cuh"
#include "ggnn/query/cuda_knn_stats_query_layer.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"
#include "ggnn/utils/cuda_knn_ggnn_results.cuh"

// only needed for getTotalSystemMemory()
#include <unistd.h>

size_t getTotalSystemMemory()
{
    size_t pages = sysconf(_SC_PHYS_PAGES);
    // this excludes memory used for caching files...
    //size_t free_pages = sysconf(_SC_AVPHYS_PAGES);
    size_t page_size  = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}


/**
 * GGNN multi-GPU wrapper
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
struct GGNNMultiGPU {

  using Dataset = Dataset<KeyT, BaseT, BAddrT>;
  using GGNNGPUInstance = GGNNGPUInstance<measure, KeyT, ValueT, GAddrT, BaseT, BAddrT, D, KBuild, KF, KQuery, S>;
  using GGNNResults = GGNNResults<measure, KeyT, ValueT, BaseT, BAddrT, KQuery>;

  Dataset dataset;

  /// one instance per GPU
  std::vector<GGNNGPUInstance> ggnn_gpu_instances;

  int num_parts {0};
  bool swap_to_disk {false};
  bool swap_to_ram {false};
  bool process_shards_back_to_front {false};
  std::string graph_dir;

  const int L;
  const float tau_build;

  const bool generate_gt;

  GGNNMultiGPU(const std::string& basePath, const std::string& queryPath,
       const std::string& gtPath, const int L, const float tau_build, const size_t N_base = std::numeric_limits<size_t>::max())
      : dataset{basePath, queryPath, gtPath, N_base},
        L{L},
        tau_build{tau_build},
        generate_gt{gtPath.empty()} {
    CHECK_EQ(dataset.D, D) << "DIM needs to be the same";
  }

  void ggnnMain(const std::vector<int>& gpus, const std::string& mode,
                const int N_shard, const std::string& graph_dir,
                const int refinement_iterations,
                const bool grid_search) {

    const bool build = mode.find('b') != std::string::npos;
    const bool store =  build && mode.find('s') != std::string::npos;
    const bool load  = !build && mode.find('l') != std::string::npos;
    const bool query = mode.find('q') != std::string::npos;

    {
      std::string mode("Mode: ");
      if (build)
        mode += "BUILD";
      else if (load)
        mode += "LOAD";
      if (store)
        mode += " AND STORE";
      if (query)
        mode += " AND QUERY";
      VLOG(0) << mode;
    }

    configure(gpus, build, N_shard, graph_dir);

    if (build) {
      this->build(refinement_iterations);
      if (store)
        this->store();
    }
    else if (load)
      this->load();
    if (query) {
      if (grid_search) {
        for (int i=0; i<70; ++i)
          this->query(i*0.01f);
        for (int i=7; i<=20; ++i)
          this->query(i*0.1f);
      }
      else {
        this->query(0.3f);
        this->query(0.4f);
        this->query(0.5f);
        this->query(0.6f);
      }
    }
  }

  static size_t computeGraphSize(const int N_shard, const int L) {
    /// theoretical growth factor (number of sub-graphs merged together per
    /// layer)
    const float growth = powf(N_shard / static_cast<float>(S), 1.f / (L - 1));

    const int Gf = growth;
    const int Gc = growth + 1;

    const float S0f = N_shard / (pow(Gf, (L - 1)));
    const float S0c = N_shard / (pow(Gc, (L - 1)));

    const bool is_floor =
        (growth > 0) && ((S0c < KBuild) || (fabs(S0f - S) < fabs(S0c - S)));

    const int G = (is_floor) ? Gf : Gc;
    const int S0 = (is_floor) ? S0f : S0c;
    const int S0_off = N_shard - pow(G, L - 1) * S0;

    int N_all = 0;
    int ST_all = 0;

    int N_current = N_shard;
    for (int l = 0; l < L; l++) {
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

    // just to make sure that everything is sufficiently aligned
    auto align8 = [](size_t size) -> size_t {return ((size+7)/8)*8;};

    const size_t graph_size = align8(static_cast<size_t>(N_all) * KBuild * sizeof(KeyT));
    const size_t selection_translation_size = align8(ST_all * sizeof(KeyT));
    // const size_t nn1_dist_buffer_size = N * sizeof(ValueT);
    const size_t nn1_stats_size = align8(2 * sizeof(ValueT));
    const size_t total_graph_size = graph_size + 2 * selection_translation_size + nn1_stats_size;

    return total_graph_size;
  }

  void configure(const std::vector<int>& gpu_ids={0}, bool enable_construction=true,
                 int N_shard=-1, const std::string graph_dir="") {
    ggnn_gpu_instances.clear();

    CHECK(!graph_dir.empty());
    if (graph_dir.back() == '/')
      this->graph_dir = graph_dir;
    else
      this->graph_dir = graph_dir+'/';

    const int num_gpus = gpu_ids.size();
    // determine shard sizes and number of iterations
    if (N_shard < 0)
      N_shard = dataset.N_base/num_gpus;
    const int num_iterations = dataset.N_base/(N_shard * num_gpus);
    num_parts = num_gpus*num_iterations;
    CHECK_EQ(N_shard*num_gpus*num_iterations, dataset.N_base) << "N_shard x num_gpus xnum_iterations needs to be equal to N_base, for now.";

    // determine number of cpu-side buffers
    const size_t total_graph_size = computeGraphSize(N_shard, L);
    const size_t total_memory = getTotalSystemMemory();
    // guess the available memory (assume 1/8 used elsewhere, subtract dataset)
    const size_t available_memory = total_memory-total_memory/8-sizeof(ValueT)*static_cast<size_t>(dataset.N_base)*D;

    const int max_parts_per_gpu = available_memory/(total_graph_size*num_gpus);
    LOG(INFO) << "estimated remaining host memory (" << available_memory/(1024.0f*1024.0f*1024.0f)
              << " GB) suffices for " << max_parts_per_gpu << " parts per GPU ("
              << total_graph_size/(1024.0f*1024.0f*1024.0f) << " GB each).";

    CHECK_GT(max_parts_per_gpu, 0) << "use smaller shards.";

    const int num_cpu_buffers_per_gpu = min(num_iterations, max_parts_per_gpu);

    swap_to_disk = num_cpu_buffers_per_gpu < num_iterations;

    ggnn_gpu_instances.reserve(num_gpus);

    VLOG(4) << "allocating shards...";
    for (int device_i=0; device_i<num_gpus; ++device_i) {
      const int gpu_id = gpu_ids[device_i];
      CHECK_CUDA(cudaSetDevice(gpu_id));

      ggnn_gpu_instances.emplace_back(gpu_id, &dataset, N_shard, L, enable_construction, tau_build, num_iterations, num_cpu_buffers_per_gpu);

      swap_to_ram |= ggnn_gpu_instances.at(device_i).ggnn_shards.size() < num_iterations;

      if (!swap_to_disk) {
        const size_t num_gpu_shards = ggnn_gpu_instances.at(device_i).ggnn_shards.size();
        for (int i=0; i<num_gpu_shards; ++ i)
          ggnn_gpu_instances.at(device_i).loadShardBaseDataAsync(device_i * num_iterations + i, i);
      }
    }

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());
    VLOG(4) << "GGNN multi-GPU setup configured.";
    if (swap_to_disk)
      VLOG(4) << "shards will be swapped to disk. (not all parts fit into ram simultaneously)";
    if (swap_to_ram)
      VLOG(4) << "shards will be swapped to ram. (not all shards fit onto the gpu simultaneously)";
  }

  void build(const int refinement_iterations) {
    CHECK(!ggnn_gpu_instances.empty()) << "configure() the multi-GPU setup first!";

    const int num_gpus = int(ggnn_gpu_instances.size());
    const int N_shard = ggnn_gpu_instances[0].N_shard;
    const int num_iterations = int(num_parts/ggnn_gpu_instances.size());

    std::vector<int64_t> build_times(num_parts);
    VLOG(0) << "GGNN::build()"
            << " | num_gpus: " << num_gpus
            << " | N_shard: " << N_shard
            << " | num_iterations: " << num_iterations;

    std::vector<std::thread> threads;
    threads.reserve(num_gpus);

    for (int device_i = 0; device_i < num_gpus; device_i++) {
      std::thread t([&, device_i]() {
        auto& gpu_instance = ggnn_gpu_instances.at(device_i);
        const int gpu_id = gpu_instance.gpu_id;
        const int num_gpu_buffers = gpu_instance.ggnn_shards.size();
        const int num_cpu_buffers = gpu_instance.ggnn_cpu_buffers.size();
        CHECK_CUDA(cudaSetDevice(gpu_id));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // printf("[gpu: %d] N_shard: %d \n", gpu_id, N_shard);
        VLOG(1) << "[GPU: " << gpu_id << "] N_shard: " << N_shard;

        if (swap_to_disk) {
          for (int i = 0; i < num_gpu_buffers; i++)
            gpu_instance.loadShardBaseDataAsync(device_i * num_iterations + i, i);
        }
        if (swap_to_ram) {
          for (int i = 0; i < num_cpu_buffers; i++)
            gpu_instance.ggnn_cpu_buffers[i].current_part_id = -1;
        }

        for (int i = 0; i < num_iterations; i++)
        {
          const int part_id = device_i * num_iterations + i;

          auto& shard = gpu_instance.ggnn_shards.at(i%gpu_instance.ggnn_shards.size());

          cudaStreamSynchronize(shard.stream);

          cudaEventRecord(start, shard.stream);
          gpu_instance.build(part_id, i);

          for (int refinement_step = 0; refinement_step < refinement_iterations;
              ++refinement_step) {
            DLOG(INFO) << "Refinement step " << refinement_step;
            gpu_instance.refine(i);
          }
          cudaEventRecord(stop, shard.stream);

          cudaEventSynchronize(stop);
          float milliseconds = 0;
          cudaEventElapsedTime(&milliseconds, start, stop);
          VLOG(0) << "[GPU: " << gpu_id << "] part: " << part_id << " => seconds: " << milliseconds/1000.f << " [" << N_shard << " points build -> " << milliseconds*1000.0f/N_shard << " us/point] \n";
          build_times[part_id] = milliseconds;

          if (swap_to_disk || swap_to_ram) {
            if (swap_to_disk)
              gpu_instance.storePartAsync(graph_dir, part_id, i);
            else
              gpu_instance.downloadPartAsync(part_id, i);

            if (i+num_gpu_buffers < num_iterations)
              gpu_instance.loadShardBaseDataAsync(part_id+num_gpu_buffers, i+num_gpu_buffers);
          }
        }

        if (swap_to_disk || swap_to_ram)
        {
          for (int i = 0; i < num_iterations; i++)
            gpu_instance.waitForDiskIO(i);
        }

        VLOG(0) << "[GPU: " << gpu_id << "] build() done.";
      });
      threads.push_back(std::move(t));
    }

    for (auto&& t : threads) {
        t.join();
    }

    float build_time_ms = 0.f;
    for (auto&& b : build_times)
    {
      build_time_ms += static_cast<float>(b);
    }

    VLOG(0) << "Combined build time: " << build_time_ms/1000.f << " s \n";

    process_shards_back_to_front = true;
  }

  void store() {
    CHECK(!ggnn_gpu_instances.empty()) << "configure() the multi-GPU setup first!";
    if (swap_to_disk) {
      VLOG(4) << "graph should already be stored on-the-fly";
      return;
    }

    const int num_gpus = int(ggnn_gpu_instances.size());
    const int num_iterations = int(num_parts/ggnn_gpu_instances.size());

    std::vector<std::thread> threads;
    threads.reserve(num_gpus);

    for (int device_i = 0; device_i < num_gpus; device_i++) {
      std::thread t([&, device_i]() {
        auto& gpu_instance = ggnn_gpu_instances.at(device_i);
        const int gpu_id = gpu_instance.gpu_id;

        for (int i = 0; i < num_iterations; i++) {
          const int part_id = device_i * num_iterations + i;
          gpu_instance.storePartAsync(graph_dir, part_id, i);
        }
        for (int i = 0; i < num_iterations; i++) {
          gpu_instance.waitForDiskIO(i);
        }

        VLOG(0) << "[GPU: " << gpu_id << "] store() done.";
      });
      threads.push_back(std::move(t));
    }

    for (auto&& t : threads) {
        t.join();
    }
  }

  void load() {
    CHECK(!ggnn_gpu_instances.empty()) << "configure() the multi-GPU setup first!";
    if (swap_to_disk) {
      VLOG(4) << "graph will be loaded on-the-fly";
      return;
    }

    const int num_gpus = int(ggnn_gpu_instances.size());
    const int num_iterations = int(num_parts/ggnn_gpu_instances.size());

    std::vector<std::thread> threads;
    threads.reserve(num_gpus);

    for (int device_i = 0; device_i < num_gpus; device_i++) {
      std::thread t([&, device_i]() {
        auto& gpu_instance = ggnn_gpu_instances.at(device_i);
        const int gpu_id = gpu_instance.gpu_id;

        for (int i = 0; i < num_iterations; i++) {
          const int part_id = device_i * num_iterations + i;
          gpu_instance.loadPartAsync(graph_dir, part_id, i);
        }
        for (int i = 0; i < num_iterations; i++)
          gpu_instance.waitForDiskIO(i);

        VLOG(0) << "[GPU: " << gpu_id << "] load() done.";
      });
      threads.push_back(std::move(t));
    }

    for (auto&& t : threads) {
        t.join();
    }
  }

  void query(const float tau_query) {
    CHECK(!ggnn_gpu_instances.empty()) << "configure() the multi-GPU setup first!";

    dataset.template checkForDuplicatesInGroundTruth<measure, ValueT>(KQuery);

    const int num_gpus = int(ggnn_gpu_instances.size());
    const int N_shard = ggnn_gpu_instances[0].N_shard;
    const int num_iterations = int(num_parts/ggnn_gpu_instances.size());

    VLOG(0) << "GGNN::query()"
            << " | tau_query: " << tau_query
            << " | num_gpus: " << num_gpus
            << " | N_shard: " << N_shard
            << " | num_iterations: " << num_iterations;

    GGNNResults ggnn_results{&dataset, num_gpus, num_iterations};

    std::vector<std::thread> threads;
    threads.reserve(num_gpus);

    for (int device_i = 0; device_i < num_gpus; device_i++) {
      std::thread t([&, device_i]() {
        auto& gpu_instance = ggnn_gpu_instances.at(device_i);
        const int gpu_id = gpu_instance.gpu_id;
        const int num_gpu_buffers = gpu_instance.ggnn_shards.size();
        const int num_cpu_buffers = gpu_instance.ggnn_cpu_buffers.size();
        const int prefetch_amount = min(num_cpu_buffers, num_gpu_buffers);
        CHECK_CUDA(cudaSetDevice(gpu_id));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;

        cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaPeekAtLastError());

        if (swap_to_disk || swap_to_ram) {
          // initially, prefetch for the entire gpu
          for (int i = 0; i < num_gpu_buffers; i++) {
            const int j = process_shards_back_to_front ? num_iterations-i-1 : i;
            const int part_id = device_i * num_iterations + j;
            gpu_instance.loadPartAsync(graph_dir, part_id, j);
          }
        }

        // TODO: warmup (here or in another function?)

        for (int i = 0; i < num_iterations; i++)
        {
          const int j = process_shards_back_to_front ? num_iterations-i-1 : i;
          const int part_id = device_i * num_iterations + j;

          auto& shard = gpu_instance.ggnn_shards.at(j%gpu_instance.ggnn_shards.size());

          if (swap_to_disk || swap_to_ram) {
            auto begin = std::chrono::high_resolution_clock::now();
            gpu_instance.waitForDiskIO(j);
            auto end = std::chrono::high_resolution_clock::now();
            auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
            VLOG(0) << "[GPU: " << gpu_id << "] shard-swap delay: " << cpu_us.count()*0.001f << " ms.";
          }

          cudaStreamSynchronize(shard.stream);

          cudaEventRecord(start, shard.stream);
          gpu_instance.template queryLayer<32, 400, 448, 64>(j);
          cudaEventRecord(stop, shard.stream);

          if (swap_to_disk || swap_to_ram) {
            // start the upload for the next shard after starting the current query
            // then, it should be able to overlap
            // prefetch only as much in parallel as there are cpu buffers
            if (process_shards_back_to_front) {
              if (j-prefetch_amount < num_iterations-num_gpu_buffers && j-prefetch_amount >= 0) {
                gpu_instance.loadPartAsync(graph_dir, part_id-prefetch_amount, j-prefetch_amount);
              }
            }
            else if (j+prefetch_amount >= num_gpu_buffers && j+prefetch_amount < num_iterations) {
              gpu_instance.loadPartAsync(graph_dir, part_id+prefetch_amount, j+prefetch_amount);
            }
          }

          cudaEventSynchronize(stop);

          cudaEventElapsedTime(&milliseconds, start, stop);
          VLOG(0) << "[GPU: " << gpu_id << "] query part: " << part_id << " => ms: " << milliseconds << " [" << dataset.N_query << " points query -> " << milliseconds*1000.0f/dataset.N_query << " us/point] \n";
        }

        const cudaStream_t shard0Stream = gpu_instance.ggnn_shards.at(0).stream;

        cudaEventRecord(start, shard0Stream);
        gpu_instance.ggnn_query.sortAsync(shard0Stream);
        cudaEventRecord(stop, shard0Stream);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        if(num_iterations > 1) {
          VLOG(0) << "[GPU: " << device_i << "] query sort: " << " => ms: " << milliseconds << " [" << dataset.N_query << " points query -> " << milliseconds*1000.0f/dataset.N_query << " us/point] \n";
        }

        ggnn_results.loadAsync(gpu_instance.ggnn_query, device_i, shard0Stream);
        cudaStreamSynchronize(shard0Stream);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaPeekAtLastError());

        VLOG(0) << "[GPU: " << gpu_id << "] query() done.";
      });
      threads.push_back(std::move(t));
    }

    for (auto&& t : threads) {
        t.join();
    }

    // CPU Zone:
    ggnn_results.merge();
    ggnn_results.evaluateResults();

    // process the shards in reverse order during the next query for improved cache utilization
    process_shards_back_to_front = !process_shards_back_to_front;
  }
}; // GGNN

#endif  // INCLUDE_GGNN_CUDA_KNN_GGNN_MULTI_GPU_CUH_
