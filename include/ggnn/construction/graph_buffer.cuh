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

#ifndef INCLUDE_GGNN_GRAPH_BUFFER_CUH
#define INCLUDE_GGNN_GRAPH_BUFFER_CUH

#include <ggnn/base/def.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/dataset.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ggnn {

/**
 * GGNN graph buffer data
 * auxiliary data needed for graph construction once per GPU
 *
 * @param KeyT datatype of dataset indices
 * @param ValueT distance value type
 */
template <typename KeyT, typename ValueT>
struct GraphBuffer {
  struct PartSizes {
    PartSizes(const GraphConfig& config)
        : graph_buffer_size{align8(static_cast<size_t>(config.N) * config.KBuild * sizeof(KeyT))},
          nn1_dist_buffer_size{align8(static_cast<size_t>(config.N) * sizeof(ValueT))},
          rng_size{align8(static_cast<size_t>(config.N) * sizeof(float))},
          sym_buffer_size{align8(static_cast<size_t>(config.N) * config.KF * sizeof(KeyT))},
          sym_atomic_size{align8(static_cast<size_t>(config.N) * sizeof(uint32_t))}
    {
    }

    const size_t graph_buffer_size;
    const size_t nn1_dist_buffer_size;
    const size_t rng_size;
    const size_t sym_buffer_size;
    const size_t sym_atomic_size;

    size_t getBufferSize() const
    {
      const size_t merge_size = nn1_dist_buffer_size + graph_buffer_size;
      const size_t select_size = nn1_dist_buffer_size + rng_size;
      const size_t sym_size = sym_buffer_size + sym_atomic_size;

      // NOTE: we're ignoring the size for stats, which should be lower
      return std::max({merge_size, select_size, sym_size});
    }
  };

  /// distance to nearest known neighbor per point
  ValueT* nn1_dist_buffer{nullptr};

  // BUFFER
  KeyT* graph_buffer{nullptr};
  KeyT* sym_buffer{nullptr};

  float* rng{nullptr};

  uint32_t* sym_atomic{nullptr};

  // cub buffer
  std::byte* temp_storage_cub{nullptr};

  size_t temp_storage_bytes_cub{0};

  Dataset<std::byte> memory{};

  GraphBuffer() = default;
  GraphBuffer(const GraphConfig& graph_config, Dataset<std::byte>&& memory);

  GraphBuffer(const GraphBuffer& other) = delete;
  GraphBuffer(GraphBuffer&& other) noexcept = default;
  GraphBuffer& operator=(const GraphBuffer& other) = delete;
  GraphBuffer& operator=(GraphBuffer&& other) noexcept = default;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_GRAPH_BUFFER_CUH
