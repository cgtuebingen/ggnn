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

#include <ggnn/base/ggnn.cuh>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ggnn;

int main()
{
  // create some data
  const size_t N_base = 10'000;
  const size_t N_query = 10'000;
  const uint32_t dim = 128;

  std::vector<float> base_data(N_base * dim);
  std::vector<float> query_data(N_query * dim);

  std::default_random_engine prng{};
  std::uniform_real_distribution<float> uniform{0.0f, 1.0f};

  for (float& x : base_data) {
    x = uniform(prng);
  }
  for (float& x : query_data) {
    x = uniform(prng);
  }

  /// data type for addressing points
  using KeyT = int32_t;
  /// data type of computed distances
  using ValueT = float;
  using GGNN = GGNN<KeyT, ValueT>;

  // Initialize GGNN
  GGNN ggnn{};

  // Initialize the datasets containing the base data and query data
  Dataset<float> base = Dataset<float>::copy(base_data, dim, true);
  Dataset<float> query = Dataset<float>::copy(query_data, dim, true);

  // pass the base to GGNN as reference
  ggnn.setBaseReference(base);

  // build the search graph
  ggnn.build(24, 0.5f);

  // run the query and store indices & squared distances
  const uint32_t KQuery = 10;
  const auto [indices, dists] = ggnn.query(query, KQuery, 0.5f);

  // print the results for the first query
  std::cout << "Result for the first query vector: \n";
  for (uint32_t i = 0; i < KQuery; i++) {
    // std::cout << "Base Idx: ";
    std::cout << "Distance to vector at base[";
    std::cout.width(5);
    std::cout << indices[i];
    std::cout << "]: " << dists[i] << "\n";
  }

  return 0;
}
