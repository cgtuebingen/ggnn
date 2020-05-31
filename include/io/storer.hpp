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

#ifndef INCLUDE_IO_STORER_HPP_
#define INCLUDE_IO_STORER_HPP_

#include <fstream>
#include <string>

template <typename ValueT>
class Storer {
 public:
  Storer() {}

  explicit Storer(std::string path, uint dimension, uint num_elements)
      : path(path), dimension(dimension), num_elements(num_elements) {
    hnd = new std::ofstream(path, std::ios_base::out | std::ios_base::binary |
                                      std::ios_base::trunc);

    if (!hnd->good()) {
      hnd->close();
      throw std::runtime_error("Not able to write to path: " + path);
    }
  }

  virtual ~Storer() {
    this->hnd->close();
    delete hnd;
  }

  /**
   * load vectors
   * @param skip number of vectors to skip (not bytes, not values)
   * @param num  number of elements to read
   */
  virtual void store(ValueT *dst, size_t num) = 0;

  uint Dim() const { return dimension; }
  uint Num() const { return num_elements; }
  std::string Path() const { return path; }

 protected:
  std::string path;
  std::ofstream *hnd;

  uint dimension;
  uint num_elements;
};

#endif  // INCLUDE_IO_STORER_HPP_