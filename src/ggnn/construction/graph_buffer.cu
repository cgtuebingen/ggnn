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

#include <ggnn/construction/graph_buffer.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/fwd.h>
#include <ggnn/base/lib.h>

#include <ggnn/cuda_utils/check.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <glog/logging.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace ggnn {

template <typename KeyT, typename ValueT>
GraphBuffer<KeyT, ValueT>::GraphBuffer(const GraphConfig& graph_config, Dataset<std::byte>&& memory)
    : memory{std::move(memory)}
{
  const PartSizes buffer_part_sizes{graph_config};

  // stats
  {
    ValueT* unused{nullptr};
    size_t temp_storage_bytes_sum{0};
    size_t temp_storage_bytes_max{0};

    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_sum, unused, unused,
                           static_cast<int>(graph_config.N));
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes_max, unused, unused,
                           static_cast<int>(graph_config.N));
    temp_storage_bytes_cub = align8(std::max(temp_storage_bytes_sum, temp_storage_bytes_max));
  }

  // const size_t total_size = graph_buffer_size + sym_buffer_size + rng_size + sym_atomic_size +
  // sym_statistics_size + nn1_dist_buffer_size + temp_storage_bytes_sum + temp_storage_bytes_max;

  // this will work as long as the construction code remains as is
  const size_t merge_size =
      buffer_part_sizes.nn1_dist_buffer_size + buffer_part_sizes.graph_buffer_size;
  const size_t select_size = buffer_part_sizes.nn1_dist_buffer_size + buffer_part_sizes.rng_size;
  const size_t stats_size = buffer_part_sizes.nn1_dist_buffer_size + temp_storage_bytes_cub;
  const size_t sym_size = buffer_part_sizes.sym_buffer_size + buffer_part_sizes.sym_atomic_size;

  const size_t overlapped_size = std::max({merge_size, select_size, stats_size, sym_size});

  VLOG(2) << "GraphBuffer(): allocating GPU memory... (" << sizeInGB(overlapped_size)
          << " GB total).\n";

  CHECK_GE(this->memory.size_bytes(), overlapped_size);

  nn1_dist_buffer = reinterpret_cast<ValueT*>(this->memory.data());
  graph_buffer =
      reinterpret_cast<KeyT*>(this->memory.data() + buffer_part_sizes.nn1_dist_buffer_size);
  rng = reinterpret_cast<float*>(this->memory.data() + buffer_part_sizes.nn1_dist_buffer_size);
  temp_storage_cub = this->memory.data() + buffer_part_sizes.nn1_dist_buffer_size;
  sym_buffer = reinterpret_cast<KeyT*>(this->memory.data());
  sym_atomic = reinterpret_cast<uint32_t*>(this->memory.data() + buffer_part_sizes.sym_buffer_size);
}

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_INSTANTIATE_STRUCT, GraphBuffer);

};  // namespace ggnn
