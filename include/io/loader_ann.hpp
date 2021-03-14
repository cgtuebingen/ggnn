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

#ifndef INCLUDE_IO_LOADER_ANN_HPP_
#define INCLUDE_IO_LOADER_ANN_HPP_

#include <fstream>
#include <string>
#include "loader.hpp"

template <typename ValueT>
class XVecsLoader : public Loader<ValueT> {
 public:
  explicit XVecsLoader(const std::string& path) : Loader<ValueT>(path) {
    // find dimension
    this->hnd->seekg(0, std::ios::beg);
    this->hnd->read(reinterpret_cast<char*>(&this->dimension), sizeof(int));

    size_t stride = sizeof(uint32_t) + this->dimension * sizeof(ValueT);

    // calc file size
    this->hnd->seekg(0, std::ios::beg);
    std::streampos fsize = this->hnd->tellg();
    this->hnd->seekg(0, std::ios::end);
    fsize = this->hnd->tellg() - fsize;

    this->num_elements = fsize / stride;
    this->hnd->seekg(0, std::ios::beg);

    DLOG(INFO) << "Open " << path << " with " << this->num_elements << " "
               << this->dimension << "-dim vectors.";
  }

  void load(ValueT* dst, size_t skip, size_t num) override {
    DLOG(INFO) << "Loading " << num << " vectors starting at " << skip
               << " ...";

    size_t stride = 1 * sizeof(uint32_t) + this->dimension * sizeof(ValueT);
    this->hnd->seekg(stride * skip);

    int32_t dim;

    for (size_t n = 0; n < num; ++n) {
      // skip dimension
      this->hnd->read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
      CHECK_EQ(dim, this->dimension) << "dimension mismatch";

      this->hnd->read(reinterpret_cast<char*>(dst),
                      this->dimension * sizeof(ValueT));
      dst += this->dimension;
    }

    DLOG(INFO) << "Done";
  }
};

using FVecsLoader = XVecsLoader<float>;
using IVecsLoader = XVecsLoader<int>;
using BVecsLoader = XVecsLoader<uint8_t>;

#endif  // INCLUDE_IO_LOADER_ANN_HPP_
