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

#include <ggnn/base/def.h>
#include <ggnn/base/graph_config.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <glog/logging.h>

namespace ggnn {

constexpr uint32_t powInt(const uint32_t base, const uint32_t power)
{
  if (!power)
    return 1;
  else if (power == 1)
    return base;
  return base * powInt(base, power - 1);
}

GraphDimensions::GraphDimensions(uint32_t N, uint32_t S, uint32_t G)
{
  // fixed block hierarchy
  for (uint32_t l = L - 1, B = 1; l != -1U; --l, B *= G) {
    Bs[l] = B;
    Ns[l] = B * S;
  }
  // bottom layer has all points (block sizes adjust accordingly)
  Ns[0] = N;
  // no offsets in layer 0
  Ns_offsets[0] = 0;
  STs_offsets[0] = 0;
  // no selection/translation in layer 0
  STs_offsets[1] = 0;
  Ns_offsets[1] = N;

  for (uint32_t l = 2; l < L; ++l) {
    Ns_offsets[l] = Ns_offsets[l - 1] + Ns[l - 1];
    STs_offsets[l] = STs_offsets[l - 1] + Ns[l - 1];
  }
  N_all = Ns_offsets[L - 1] + Ns[L - 1];
  ST_all = STs_offsets[L - 1] + Ns[L - 1];
}

GraphDerivedParameters::GraphDerivedParameters(const GraphParameters& params)
    : GraphParameters{params}
{
  /// theoretical growth factor (number of sub-graphs merged together per layer)
  /// graph grows top down: 1*S, G*S, G*G*S, G*G*G*S0+S0_off == N
  const float growth = std::pow(static_cast<float>(N) / static_cast<float>(S), 1.f / (L - 1));

  // pick between the closest integers
  const uint32_t Gf = static_cast<uint32_t>(growth);
  const uint32_t Gc = Gf + 1;

  // resulting level 0 (base level) segment sizes
  const float S0f = static_cast<float>(N) / (std::pow(static_cast<float>(Gf), (L - 1.0f)));
  const float S0c = static_cast<float>(N) / (std::pow(static_cast<float>(Gc), (L - 1.0f)));

  // use the larger layer 0 segment size (S0f)
  // if the smaller one (S0c) becomes too small to establish meaningful neighborhoods within it
  // or if it (S0f) is closer to S than the smaller option (S0c)
  const bool is_floor =
      (static_cast<uint32_t>(S0c) < KBuild) ||
      (std::abs(S0f - static_cast<float>(S)) < std::abs(S0c - static_cast<float>(S)));

  G = (is_floor) ? Gf : Gc;
  S0 = (is_floor) ? static_cast<uint32_t>(S0f) : static_cast<uint32_t>(S0c);
  S0_off = N - powInt(G, L - 1) * S0;

  // parameters for selection
  SG = S / G;
  SG_off = S - SG * G;

  // TODO: can we fix that? ==> S needs to be a multiple of G
  DLOG_IF(WARNING, SG == 0) << "less than one point per segment contributes to upper level "
                               "segments. this may negatively impact search performance.";
  DLOG_IF(WARNING, SG_off > 0) << "segment's contributions to upper level segments are imbalanced. "
                                  "this may negatively impact search performance.";
}

GraphConfig::GraphConfig(const GraphParameters& params)
    : GraphDerivedParameters{params}, GraphDimensions{N, S, G}
{
  VLOG(1) << "GraphConfig(): N: " << N << ", K: " << KBuild << ", KF: " << KF << ", L: " << L
          << ", G: " << G << ", S: " << S << ", S0: " << S0 << ", S0_off: " << S0_off
          << ", SG: " << SG << ", SG_off: " << SG_off;
}

size_t GraphConfig::maxBaseAddr() const
{
  return std::max(static_cast<size_t>(N) * D, static_cast<size_t>(N_all) * KBuild);
}

};  // namespace ggnn
