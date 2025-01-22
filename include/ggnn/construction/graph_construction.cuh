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

#ifndef INCLUDE_GGNN_GRAPH_CONSTRUCTION_H
#define INCLUDE_GGNN_GRAPH_CONSTRUCTION_H

#include <ggnn/base/def.h>
#include <ggnn/base/fwd.h>

#include <memory>

namespace ggnn {

template <typename KeyT, typename ValueT, typename BaseT>
struct GPUInstance;

/**
 * Wrapper for graph construction kernels.
 */
template <typename KeyT, typename ValueT, typename BaseT>
class GraphConstruction {
 public:
  using GPUInstance = ggnn::GPUInstance<KeyT, ValueT, BaseT>;
  using Graph = ggnn::Graph<KeyT, ValueT>;

  GraphConstruction() = default;
  GraphConstruction(GPUInstance& gpu_instance, float tau_build, const DistanceMeasure measure);
  virtual ~GraphConstruction() = default;
  GraphConstruction(const GraphConstruction&) = delete;
  GraphConstruction(GraphConstruction&&) noexcept = default;
  GraphConstruction& operator=(const GraphConstruction&) = delete;
  GraphConstruction& operator=(GraphConstruction&&) noexcept = default;

  virtual void build(Graph& graph, const Dataset<BaseT>& base, const cudaStream_t stream)
  {
    pimpl->build(graph, base, stream);
  }
  virtual void refine(Graph& graph, const Dataset<BaseT>& base, const cudaStream_t stream)
  {
    pimpl->refine(graph, base, stream);
  }

 private:
  std::unique_ptr<GraphConstruction> pimpl;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_GRAPH_CONSTRUCTION_H
