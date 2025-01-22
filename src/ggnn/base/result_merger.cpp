/* Copyright 2025 ComputerGraphics Tuebingen. All Rights Reserved.

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
// converted to GGNN library by: Lukas Ruppert, Deborah Kornwolf

#include <ggnn/base/result_merger.h>
#include <ggnn/base/dataset.cuh>

#include <ggnn/base/lib.h>

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace ggnn {

template <typename KeyT, typename ValueT>
ResultMerger<KeyT, ValueT>::ResultMerger(const uint32_t N_query, const uint32_t KQuery,
                                         const uint32_t num_gpus, const uint32_t num_shards_per_gpu)
    : N_query{N_query}, KQuery{KQuery}, num_gpus{num_gpus}, num_shards_per_gpu{num_shards_per_gpu}
{
  CHECK_GE(num_gpus, 1);
  CHECK_GE(num_shards_per_gpu, 1);

  partial_results_per_gpu.reserve(num_gpus);
  for (uint32_t i = 0; i < num_gpus; ++i) {
    partial_results_per_gpu.emplace_back(
        Results{Dataset<KeyT>::empty(N_query, KQuery * num_shards_per_gpu, true),
                Dataset<ValueT>::empty(N_query, KQuery * num_shards_per_gpu, true)});
  }
}

template <typename KeyT, typename ValueT>
typename ResultMerger<KeyT, ValueT>::Results ResultMerger<KeyT, ValueT>::merge(uint32_t N_shard) &&
{
  // for one part on one GPU, the results are directly passed through.
  if (num_gpus == 1 && num_shards_per_gpu == 1) {
    return std::move(partial_results_per_gpu.at(0));
  }

  Results merged_results = {Dataset<KeyT>::empty(N_query, KQuery),
                            Dataset<ValueT>::empty(N_query, KQuery)};

  if (num_gpus == 1) {
    // results have already been pre-sorted per GPU, so we can just copy this over
    for (uint32_t n = 0; n < N_query; n++) {
      std::copy_n(partial_results_per_gpu.at(0).ids.data() +
                      static_cast<size_t>(n) * KQuery * num_shards_per_gpu,
                  KQuery, merged_results.ids.data() + static_cast<size_t>(n) * KQuery);
      std::copy_n(partial_results_per_gpu.at(0).dists.data() +
                      static_cast<size_t>(n) * KQuery * num_shards_per_gpu,
                  KQuery, merged_results.dists.data() + static_cast<size_t>(n) * KQuery);
    }
    return merged_results;
  }

  const uint32_t stride = KQuery * num_shards_per_gpu;

  auto start = std::chrono::steady_clock::now();

  auto mergeResultPart = [&](uint32_t begin, uint32_t end) -> void {
    struct KeyDistPartition {
      KeyT key;
      ValueT dist;
      uint32_t partition;

      KeyDistPartition(KeyT key, ValueT dist, uint32_t partition)
          : key(key), dist(dist), partition(partition)
      {
      }
    };
    auto compare_heap = [](const KeyDistPartition& a, const KeyDistPartition& b) -> bool {
      return a.dist >= b.dist;
    };

    std::vector<uint32_t> part_offsets(num_gpus, 1);

    std::vector<KeyDistPartition> heap;
    heap.reserve(num_gpus);
    for (uint32_t n = begin; n < end; ++n) {
      heap.clear();
      std::fill(part_offsets.begin(), part_offsets.end(), 1);
      // fill heap with min per partition
      for (uint32_t device_i = 0; device_i < num_gpus; ++device_i) {
        const size_t pos = static_cast<size_t>(n) * stride;
        heap.emplace_back(partial_results_per_gpu.at(device_i).ids[pos],
                          partial_results_per_gpu.at(device_i).dists[pos], device_i);
      }
      std::make_heap(heap.begin(), heap.end(), compare_heap);
      // Pop min and insert from popped partition until full.
      // We can safely assume not to run out of bounds within each partition,
      // since there are as many results per part as total results requested.
      for (uint32_t k = 0; k < KQuery; ++k) {
        const KeyDistPartition top = heap.front();
        // each GPU only knows about its part of the base
        // increase the base index by the number of base points assigned to previous devices
        merged_results.ids[n * KQuery + k] =
            static_cast<KeyT>(top.partition * num_shards_per_gpu * N_shard) + top.key;
        merged_results.dists[n * KQuery + k] = top.dist;
        if (k == KQuery - 1)
          break;

        std::pop_heap(heap.begin(), heap.end(), compare_heap);
        heap.pop_back();
        const size_t pos = static_cast<size_t>(n) * stride + part_offsets[top.partition];
        ++part_offsets[top.partition];
        heap.emplace_back(partial_results_per_gpu.at(top.partition).ids[pos],
                          partial_results_per_gpu.at(top.partition).dists[pos], top.partition);
        std::push_heap(heap.begin(), heap.end(), compare_heap);
      }
    }
  };
  std::vector<std::thread> mergeThreads;

  uint32_t num_threads = std::min(N_query, std::thread::hardware_concurrency());
  uint32_t elements_per_bin = (N_query + num_threads - 1) / num_threads;
  mergeThreads.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i) {
    mergeThreads.emplace_back(mergeResultPart, i * elements_per_bin,
                              std::min(N_query, (i + 1) * elements_per_bin));
  }
  for (auto&& t : mergeThreads) {
    t.join();
  }

  auto end = std::chrono::steady_clock::now();
  auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  VLOG(0) << "[CPU] partial merge completed. " << cpu_ms.count() << " ms.";

  return merged_results;
}

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_INSTANTIATE_STRUCT, ResultMerger);

};  // namespace ggnn
