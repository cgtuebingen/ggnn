/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.

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

#ifndef INCLUDE_GGNN_CUDA_KNN_CONSTANTS_CUH_
#define INCLUDE_GGNN_CUDA_KNN_CONSTANTS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_knn_utils.cuh"

static constexpr int MAX_LAYER = 20;

__constant__ int c_Ns[MAX_LAYER];
__constant__ int c_Ns_offsets[MAX_LAYER];
__constant__ int c_STs_offsets[MAX_LAYER];

__constant__ int c_G;
__constant__ int c_L;

__constant__ float c_tau_build;
__constant__ float c_tau_query;

__constant__ int c_S0;
__constant__ int c_S0_offset;

struct ConstantInfoKernel {
  void launch() {
    printf("launch ConstantInfoKernel devId: %d L: %d \n", dev_id, L);
    launcher<<<L, 1>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    const int l = blockIdx.x;
    if (!threadIdx.x) {
      printf(
          "l: %d dev: %d -> N: %d | Noff: %d | SToff: %d "
          " | G: %d | L: %d | S0: %d S0_offset: %d\n",
          l, dev_id, c_Ns[l], c_Ns_offsets[l], c_STs_offsets[l], c_G, c_L, c_S0,
          c_S0_offset);
    }
  }

  int L;
  int dev_id;
};

#endif  // INCLUDE_GGNN_CUDA_KNN_CONSTANTS_CUH_
