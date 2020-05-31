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
// Authors: Fabian Groh, Lukas Rupert, Patrick Wieschollek, Hendrik P.A. Lensch

#ifndef INCLUDE_IO_STORER_ANN_HPP_
#define INCLUDE_IO_STORER_ANN_HPP_

#include <fstream>
#include <string>
#include "storer.hpp"

template <typename ValueT>
class XVecsStorer : public Storer<ValueT> {
 public:
  explicit XVecsStorer(std::string path, uint dimension, uint num_elements)
      : Storer<ValueT>(path, dimension, num_elements) {}

  void store(ValueT *dst, size_t num) override {
    for (uint n = 0; n < num; ++n) {
      this->hnd->write(reinterpret_cast<char *>(&this->dimension), sizeof(int));
      for (uint i = 0; i < this->dimension; ++i) {
        this->hnd->write(
            reinterpret_cast<char *>(&dst[n * this->dimension + i]),
            sizeof(ValueT));
      }
    }
  }
};

using FVecsStorer = XVecsStorer<float>;
using IVecsStorer = XVecsStorer<int>;

#endif  // INCLUDE_IO_STORER_ANN_HPP_