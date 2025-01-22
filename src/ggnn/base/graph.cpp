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

#include <ggnn/base/graph.h>

#include <ggnn/base/def.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/lib.h>
#include <ggnn/base/dataset.cuh>

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace ggnn {

template <typename KeyT, typename ValueT>
Graph<KeyT, ValueT>::Graph(const GraphConfig& graph_config, Dataset<std::byte>&& memory)
    : memory{std::move(memory)}
{
  const PartSizes graph_part_sizes{graph_config};
  const size_t total_graph_size{graph_part_sizes.getGraphSize()};

  VLOG(2) << "Graph(): N: " << graph_config.N << ", K: " << graph_config.KBuild
          << ", N_all: " << graph_config.N_all << ", ST_all: " << graph_config.ST_all << " ("
          << sizeInGB(total_graph_size) << " GB total, " << this->memory.location << ")\n";

  CHECK_GE(this->memory.size_bytes(), total_graph_size);

  const bool on_gpu = this->memory.isGPUAccessible();

  Dataset<KeyT> neighborhood_data =
      on_gpu ? Dataset<KeyT>::referenceGPUData(reinterpret_cast<KeyT*>(this->memory.data()),
                                               graph_config.N_all, graph_config.KBuild,
                                               this->memory.gpu_id)
             : Dataset<KeyT>::referenceCPUData(reinterpret_cast<KeyT*>(this->memory.data()),
                                               graph_config.N_all, graph_config.KBuild);
  Dataset<KeyT> selection_translation_data =
      on_gpu ? Dataset<KeyT>::referenceGPUData(
                   reinterpret_cast<KeyT*>(this->memory.data() + neighborhood_data.size_bytes()),
                   graph_config.ST_all * 2, 1, this->memory.gpu_id)
             : Dataset<KeyT>::referenceCPUData(
                   reinterpret_cast<KeyT*>(this->memory.data() + neighborhood_data.size_bytes()),
                   graph_config.ST_all * 2, 1);

  Dataset<KeyT> graph_layers = neighborhood_data.referenceRange(0, graph_config.N_all);
  Dataset<KeyT> translation_layers =
      selection_translation_data.referenceRange(0, graph_config.ST_all);
  Dataset<KeyT> selection_layers =
      selection_translation_data.referenceRange(graph_config.ST_all, graph_config.ST_all);

  for (uint32_t layer = 0; layer < GraphConfig::L; ++layer) {
    graph[layer] =
        graph_layers.referenceRange(graph_config.Ns_offsets[layer], graph_config.Ns[layer]);
    if (layer) {
      selection[layer] =
          selection_layers.referenceRange(graph_config.STs_offsets[layer], graph_config.Ns[layer]);
      translation[layer] = translation_layers.referenceRange(graph_config.STs_offsets[layer],
                                                             graph_config.Ns[layer]);
    }
  }

  nn1_stats =
      on_gpu ? Dataset<ValueT>::referenceGPUData(
                   reinterpret_cast<ValueT*>(this->memory.data() + neighborhood_data.size_bytes() +
                                             selection_translation_data.size_bytes()),
                   2, 1, this->memory.gpu_id)
             : Dataset<ValueT>::referenceCPUData(
                   reinterpret_cast<ValueT*>(this->memory.data() + neighborhood_data.size_bytes() +
                                             selection_translation_data.size_bytes()),
                   2, 1);

  CHECK_EQ(neighborhood_data.size_bytes() + selection_translation_data.size_bytes() +
               nn1_stats.size_bytes(),
           total_graph_size);
}

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_INSTANTIATE_STRUCT, Graph);

};  // namespace ggnn
