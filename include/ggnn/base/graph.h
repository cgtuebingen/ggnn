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
// Authors: Fabian Groh, Lukas Rupert, Patrick Wieschollek, Hendrik P.A. Lensch
// converted to GGNN library by: Lukas Ruppert, Deborah Kornwolf

#ifndef INCLUDE_GGNN_GRAPH_H
#define INCLUDE_GGNN_GRAPH_H

#include <ggnn/base/def.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/dataset.cuh>

#include <array>
#include <cstddef>

namespace ggnn {

/**
 * GGNN graph data (on the CPU)
 *
 * @param KeyT datatype of dataset indices
 * @param ValueT distance value type
 */
template <typename KeyT, typename ValueT>
struct Graph {
  struct PartSizes {
    PartSizes(const GraphConfig& config)
        : graph_size{align8(static_cast<size_t>(config.N_all) * config.KBuild * sizeof(KeyT))},
          selection_translation_size{align8(static_cast<size_t>(config.ST_all) * sizeof(KeyT))},
          // const size_t nn1_dist_buffer_size = N * sizeof(ValueT);
          nn1_stats_size{align8(2UL * sizeof(ValueT))}
    {
    }

    const size_t graph_size;
    const size_t selection_translation_size;
    const size_t nn1_stats_size;

    size_t getGraphSize() const
    {
      return graph_size + 2 * selection_translation_size + nn1_stats_size;
    }
  };

  /// neighborhood vectors
  std::array<Dataset<KeyT>, GraphConfig::L> graph{};
  /// translation of upper layer points into lowest layer
  std::array<Dataset<KeyT>, GraphConfig::L> translation{};
  /// translation of upper layer points into one layer below
  std::array<Dataset<KeyT>, GraphConfig::L> selection{};

  /// average and maximum distance to nearest known neighbors
  Dataset<ValueT> nn1_stats{};

  /// combined memory pool
  Dataset<std::byte> memory{};

  Graph() = default;
  Graph(const GraphConfig& graph_config, Dataset<std::byte>&& memory);
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_GRAPH_H
