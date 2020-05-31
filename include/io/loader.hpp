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

#ifndef INCLUDE_IO_LOADER_HPP_
#define INCLUDE_IO_LOADER_HPP_

#include <fstream>
#include <iostream>
#include <string>

template <typename ValueT>
class Loader {
 public:
  Loader() : dimension(0), num_elements(0) {}

  explicit Loader(const std::string& path) : path(path) {
    hnd = new std::ifstream(path, std::ios_base::in | std::ios_base::binary);

    if (!hnd->good()) {
      hnd->close();
      throw std::runtime_error("Dataset file " + path + " does not exists");
    }
  }

  virtual ~Loader() {
    try {
      if (hnd->is_open()) {
        this->hnd->close();
      }
      delete hnd;
    } catch (...) {
      std::cout << "could not close \n";
    }
  }

  /**
   * load vectors
   * @param skip number of vectors to skip (not bytes, not values)
   * @param num  number of elements to read
   */
  virtual void load(ValueT *dst, size_t skip, size_t num) = 0;

  int32_t Dim() const { return dimension; }
  int32_t Num() const { return num_elements; }
  std::string Path() const { return path; }

 protected:
  std::string path;
  std::ifstream *hnd;

  int32_t dimension;
  int32_t num_elements;
};

#endif  // INCLUDE_IO_LOADER_HPP_
