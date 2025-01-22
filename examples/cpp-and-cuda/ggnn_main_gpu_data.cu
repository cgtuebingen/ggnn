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

#include <cuda_runtime.h>
#include <curand.h>

using namespace ggnn;
int main()
{
  using GGNN = ggnn::GGNN<int32_t, float>;

  // create data on the GPU
  size_t N_base {10'000};
  size_t N_query {10'000};
  uint32_t D {128};

  float* base;
  float* query;

  cudaMalloc(&base, N_base * D * sizeof(float));
  cudaMalloc(&query, N_query * D * sizeof(float));

  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

  curandGenerateUniform(generator, base, N_base * D);
  curandGenerateUniform(generator, query, N_query * D);

  // initialize GGNN
  GGNN ggnn{};

  // Set the data on the GPU as the base dataset on which the graph should be built on.
  // To reference existing data, specify its pointer, the number of base vectors N_base,
  // the dimensionality of base vectors D and the gpu_id of the GPU containing the data.
  int32_t gpu_id = 0;
  ggnn.setBase(ggnn::Dataset<float>::referenceGPUData(base, N_base, D, gpu_id));

  // reference the query data which already exists on the gpu
  ggnn::Dataset<float> d_query = ggnn::Dataset<float>::referenceGPUData(query, N_query, D, 0);

  // build the search graph
  const uint32_t KBuild = 24;
  const float tau_build = 0.5f;
  ggnn.build(KBuild, tau_build);

  // run the query and store indices & distances
  const int32_t KQuery = 10;
  const auto [indices, dists] = ggnn.query(d_query, KQuery, 0.5f);

  // print the results for the first query
  std::cout << "Result for the first query verctor: \n";
  for (uint32_t i = 0; i < KQuery; i++) {
    // std::cout << "Base Idx: ";
    std::cout << "Distance to vector at base[";
    std::cout.width(5);
    std::cout << indices[i];
    std::cout << "]: " << dists[i] << "\n";
  }

  // cleanup
  curandDestroyGenerator(generator);
  cudaFree(base);
  cudaFree(query);

  return 0;
}
