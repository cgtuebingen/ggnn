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

#ifndef INCLUDE_GGNN_RESULT_MERGER_H
#define INCLUDE_GGNN_RESULT_MERGER_H

#include <ggnn/base/def.h>
#include <ggnn/base/dataset.cuh>

#include <cstdint>

#include <vector>

namespace ggnn {

template <typename KeyT, typename ValueT>
struct ResultMerger {
  using Results = ggnn::Results<KeyT, ValueT>;

  uint32_t N_query{0};
  uint32_t KQuery{0};

  uint32_t num_gpus{1};
  uint32_t num_shards_per_gpu{1};

  ResultMerger() = default;
  ResultMerger(const uint32_t N_query, const uint32_t KQuery, const uint32_t num_gpus = 1,
               const uint32_t num_shards_per_gpu = 1);

  // intermediate results per GPU, to be merged
  std::vector<Results> partial_results_per_gpu;

  /// merge together the results from multiple GPUs
  [[nodiscard]] Results merge(uint32_t N_shard) &&;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_RESULT_MERGER_H
