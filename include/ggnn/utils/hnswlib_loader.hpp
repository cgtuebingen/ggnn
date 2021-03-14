/* Copyright 2021 ComputerGraphics Tuebingen. All Rights Reserved.

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

#ifndef HNSWLIB_LOADER_HPP_
#define HNSWLIB_LOADER_HPP_

#include <gflags/gflags.h>
#include <inttypes.h>

#include <fstream>
#include <string>
#include <vector>

/**
 * HNSW loader (for known parameters only to allow for easy access to the data)
 * @param ValueT datatype of the dataset, e.g. char, int, float
 * @param D dimension of the dataset
 * @param M maximum number of neighbors in the graph = K/2
 *
 * based on Hnswlib https://github.com/nmslib/hnswlib.git
 * commit bbddf198ffc607e321a65fad159cd4b984da651b
 */
template <typename ValueT, size_t D, size_t M>
struct HNSWLoader {
  struct __attribute__((__packed__)) HNSWLevel0Element {
    uint16_t link_count;
    uint16_t padding;
    uint32_t links[M * 2];
    ValueT base_vector[D];
    size_t label;  // unaligned
  };

  /*
  // not relevant for GGNN-query
  struct HNSWUpperLevelElement {
    uint16_t link_count;
    uint16_t padding;
    uint32_t links[M];
  };
  */

  struct HNSWHeader {  // based on HierarchicalNSW::saveIndex()
    size_t offsetLevel0_;
    size_t max_elements_;
    size_t cur_element_count;
    size_t size_data_per_element_;
    size_t label_offset_;
    size_t offsetData_;
    int32_t maxlevel_;
    uint32_t enterpoint_node_;
    size_t maxM_;
    size_t maxM0_;
    size_t M_;
    double mult_;
    size_t ef_construction_;

    bool verify() {
      CHECK_EQ(M_, M);
      CHECK_EQ(maxM0_, 2 * M_);
      const size_t size_links_level0_ =
          maxM0_ * sizeof(uint32_t) + sizeof(uint32_t);
      const size_t data_size_ = D * sizeof(ValueT);
      CHECK_EQ(size_data_per_element_,
               size_links_level0_ + data_size_ + sizeof(size_t));
      CHECK_EQ(size_data_per_element_, sizeof(HNSWLevel0Element));
      return true;
    }
  } hnsw_header;

  std::vector<HNSWLevel0Element> data_level0_memory_;  // [N];

  /*
  // not relevant for GGNN-query
  std::vector<int32_t> element_levels; // [N]
  std::vector<HNSWUpperLevelElement> linkLists_; // [sum(element_levels)]
  */

  HNSWLoader(const std::string& filename) {
    // open, verify, load L0, done. (ignore upper levels)
    std::ifstream hnsw_index_file(filename,
                                  std::ios_base::in | std::ios_base::binary);
    CHECK(hnsw_index_file.is_open());

    hnsw_index_file.seekg(0, std::ios_base::end);
    size_t filesize = hnsw_index_file.tellg();
    hnsw_index_file.seekg(0, std::ios_base::beg);

    CHECK_GT(filesize, sizeof(HNSWHeader));

    hnsw_index_file.read(reinterpret_cast<char*>(&hnsw_header),
                         sizeof(HNSWHeader));

    CHECK(hnsw_header.verify());
    CHECK_GE(filesize, sizeof(HNSWHeader) + sizeof(HNSWLevel0Element) *
                                                hnsw_header.cur_element_count);

    data_level0_memory_.resize(hnsw_header.cur_element_count);

    hnsw_index_file.read(
        reinterpret_cast<char*>(data_level0_memory_.data()),
        sizeof(HNSWLevel0Element) * hnsw_header.cur_element_count);

    hnsw_index_file.close();

    LOG(INFO) << "read HNSW base layer containing "
              << hnsw_header.cur_element_count << " elements.";
  }
};

#endif  // HNSWLIB_LOADER_HPP_
