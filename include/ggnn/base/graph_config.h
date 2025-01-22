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

#ifndef INCLUDE_GGNN_GRAPH_CONFIG_H
#define INCLUDE_GGNN_GRAPH_CONFIG_H

#include <ggnn/base/def.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ggnn {

/**
 * User-definable graph parameters
 */
struct GraphParameters {
  /// number of base points per shard
  uint32_t N{};
  /// number of dimensions in the dataset and query
  uint32_t D{};

  /// number of neighbors per point
  uint32_t KBuild{};

  /// number of layers
  static constexpr uint32_t L =
      4;  // we empirically found 4 layers to work best across all datasets
};

/**
 * Automatically derived secondary graph parameters
 */
struct GraphDerivedParameters : public GraphParameters {
  GraphDerivedParameters() = default;
  GraphDerivedParameters(const GraphParameters& params);

  /// number of inverse (foreign) links per point, part of KBuild
  uint32_t KF{KBuild / 2};

  /// growth factor (number of sub-graphs merged together per layer)
  uint32_t G{};

  /// segment size
  uint32_t S{next_multiple<uint32_t, 32U>(KF + 1)};
  /// segment size in base layer
  uint32_t S0{};
  /// number of segments in base layer with one additional element
  uint32_t S0_off{};

  /// number of points per segment selected into upper-level segment
  uint32_t SG{};
  /// number of segments per layer contributing an additional point into the upper-level segment
  uint32_t SG_off{};
};

/**
 * Automatically derived graph dimensions
 */
struct GraphDimensions {
  static constexpr uint32_t L = GraphParameters::L;

  GraphDimensions() = default;
  GraphDimensions(uint32_t N, uint32_t S, uint32_t G);

  /// total number of neighborhoods in the graph
  uint32_t N_all{};
  /// total number of selection/translation entries
  uint32_t ST_all{};

  /// blocks/segments per layer
  std::array<uint32_t, L> Bs{};  // [L]
  /// neighborhoods per layer
  std::array<uint32_t, L> Ns{};  // [L]
  /// start of neighborhoods per layer
  std::array<uint32_t, L> Ns_offsets{};  // [L]
  /// start of selection/translation per layer
  std::array<uint32_t, L> STs_offsets{};  // [L]
};

/**
 * Combined Configuration of the GGNN search graph layout
 */
struct GraphConfig : public GraphDerivedParameters, public GraphDimensions {
  using GraphParameters::L;

  GraphConfig() = default;
  GraphConfig(const GraphParameters& params);

  size_t getBaseSize(const uint32_t base_t_size) const
  {
    return align8(static_cast<size_t>(N) * D * base_t_size);
  }

  size_t maxBaseAddr() const;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_GRAPH_CONFIG_H
