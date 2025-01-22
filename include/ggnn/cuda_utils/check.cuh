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

#ifndef INCLUDE_GGNN_CHECK_CUH
#define INCLUDE_GGNN_CHECK_CUH

#include <cuda_runtime.h>

#include <glog/logging.h>

namespace ggnn {

#define CHECK_CUDA(instruction)                                                   \
  {                                                                               \
    const cudaError res = instruction;                                            \
    CHECK_EQ(res, cudaSuccess) << #instruction << " " << cudaGetErrorString(res); \
  }

};  // namespace ggnn

#endif  // INCLUDE_GGNN_CHECK_CUH
