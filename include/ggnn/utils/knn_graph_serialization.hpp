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

#ifndef KNN_GRAPH_SERIALIZATION_HPP_
#define KNN_GRAPH_SERIALIZATION_HPP_

#include <inttypes.h>
#include <fstream>
#include <type_traits>

template <typename KeyT, typename ValueT>
struct KNNGraphHeader {
  enum DataType : uint16_t {
    E_float = 0,
    E_uint8_t = 1,
    E_int32_t = 2
    // extend as needed
  };

  template <typename T>
  using ExpectedType = std::integral_constant<
      DataType,
      std::conditional<std::is_same<T, float>::value,
                       std::integral_constant<DataType, E_float>,
                       typename std::conditional<
                           std::is_same<T, uint8_t>::value,
                           std::integral_constant<DataType, E_uint8_t>,
                           typename std::conditional<
                               std::is_same<T, int32_t>::value,
                               std::integral_constant<DataType, E_int32_t>,
                               void>::type>::type>::type::value>;

  typedef ExpectedType<KeyT> ExpectedKeyType;
  typedef ExpectedType<ValueT> ExpectedValueType;

  DataType KeyType;
  DataType ValueType;
  int N, D, K, KF, L, S;
  float tau_build;

  // zero-terminated string (should contain base-file name, unused bytes should
  // be filled with 0)
  char description[256];

  KNNGraphHeader() = default;

  explicit KNNGraphHeader(int N, int D, int K, int KF, int L, int S,
                          float tau_build, const char* description)
      : KeyType{ExpectedKeyType::value},
        ValueType{ExpectedValueType::value},
        N{N},
        D{D},
        K{K},
        KF{KF},
        L{L},
        S{S},
        tau_build{tau_build} {
    // cut the beginning of the string if necessary
    if (strlen(description) > sizeof(this->description))
      description += strlen(description) - sizeof(this->description) - 1;

    strncpy(this->description, description, sizeof(this->description));
    // just to be sure
    this->description[sizeof(this->description) - 1] = 0;
  }

  bool verifyTypes() const {
    if (this->KeyType != KNNGraphHeader<KeyT, ValueT>::ExpectedKeyType::value) {
      std::cout << "key type does not match" << std::endl;
      return false;
    }
    if (this->ValueType !=
        KNNGraphHeader<KeyT, ValueT>::ExpectedValueType::value) {
      std::cout << "value type does not match" << std::endl;
      return false;
    }
    return true;
  }

  bool verify(int N, int D, int K, int KF, int L, int S, float tau_build) const {
    if (!verifyTypes()) return false;
    if (this->N != N) {
      std::cout << "N does not match" << std::endl;
      return false;
    }
    if (this->D != D) {
      std::cout << "D does not match" << std::endl;
      return false;
    }
    if (this->K != K) {
      std::cout << "K does not match" << std::endl;
      return false;
    }
    if (this->KF != KF) {
      std::cout << "KF does not match" << std::endl;
      return false;
    }
    if (this->L != L) {
      std::cout << "L does not match" << std::endl;
      return false;
    }
    if (this->S != S) {
      std::cout << "K does not match" << std::endl;
      return false;
    }
    if (this->tau_build != tau_build) {
      std::cout << "tau_build does not match" << std::endl;
      return false;
    }
    return true;
  }
};

template <typename KeyT, typename ValueT>
struct KNNGraphData {
  KNNGraphHeader<KeyT, ValueT> header;

  size_t computeSTAll() const {
    const float growth =
        powf((float)header.N / (float)header.S, 1.f / (header.L - 1));

    const int Gf = growth;
    const int Gc = growth + 1;

    const float S0f = header.N / (pow(Gf, (header.L - 1)));
    const float S0c = header.N / (pow(Gc, (header.L - 1)));

    const bool is_floor =
        (growth > 0) &&
        ((S0c < header.K) || (fabs(S0f - header.S) < fabs(S0c - header.S)));

    const int G = (is_floor) ? Gf : Gc;

    int N_all = 0;
    for (int l = 0; l < header.L; l++) {
      int N_current = (!l) ? header.N : header.S * powf(G, header.L - l - 1);
      N_all += N_current;
    }

    return N_all - header.N;
  }

  size_t sizeofGraph() const {
    return static_cast<size_t>(header.N) * static_cast<size_t>(header.K) *
           sizeof(ValueT);
  };

  size_t sizeofTranslation() const { return computeSTAll() * sizeof(KeyT); };

  size_t sizeofSelection() const { return computeSTAll() * sizeof(KeyT); };

  size_t sizeofNN1Stats() const { return 2 * sizeof(ValueT); };

  size_t sizeofNN1DistBuffer() const {
    return static_cast<size_t>(header.N) * sizeof(ValueT);
  };

  size_t totalSize() const {
    return sizeof(header) + sizeofGraph() + sizeofTranslation() +
           sizeofSelection() + sizeofNN1Stats() + sizeofNN1DistBuffer();
  }
};

template <typename KeyT, typename ValueT>
class KNNGraphReader : private KNNGraphData<KeyT, ValueT> {
  std::ifstream inFile;

 public:
  bool open(const std::string& filename) {
    if (inFile.is_open()) {
      std::cout
          << "KNNGraphReader::open(): A KNN Graph file was already opened."
          << std::endl;
      inFile.close();
    }

    inFile.open(filename, std::ifstream::in | std::ifstream::binary);
    if (!inFile.is_open()) {
      std::cout << "failed to open KNN Graph file for reading " << filename
                << std::endl;
      return false;
    }

    inFile.seekg(0, std::ios_base::end);
    size_t filesize = inFile.tellg();
    inFile.seekg(0, std::ios_base::beg);

    if (filesize < sizeof(this->header)) {
      std::cout << "invalid KNN Graph file (too small to contain header!) "
                << filename << std::endl;
      return false;
    }

    inFile.read(reinterpret_cast<char*>(&this->header), sizeof(this->header));

    if (!inFile) {
      std::cout << "io error in KNN Graph file (tried to read header) "
                << filename << std::endl;
      inFile.close();
      return false;
    }

    if (!this->header.verifyTypes()) {
      inFile.close();
      return false;
    }

    if (filesize != this->totalSize()) {
      inFile.close();
      std::cout << "invalid KNN Graph file (file size does not match the "
                   "header information!) "
                << filename << std::endl;
      return false;
    }

    std::cout << "opened KNN Graph file " << filename << std::endl;
    std::cout << "description: "
              << std::string(this->header.description,
                             std::min(strlen(this->header.description), 256UL))
              << std::endl;

    return true;
  }

  bool verify(int N, int D, int K, int KF, int L, int S, float tau_build) const {
    return this->header.verify(N, D, K, KF, L, S, tau_build);
  }

  void close() {
    if (inFile.is_open())
      inFile.close();
    else
      std::cout << "KNNGraphReader::close(): file was not open!" << std::endl;
  }

  bool read(KeyT* graph, KeyT* translation, KeyT* selection, ValueT* nn1_stats,
            ValueT* nn1_dist_buffer) {
    if (!inFile.is_open()) {
      std::cout << "KNNGraphReader::read(): file is not open!" << std::endl;
      return false;
    }

    inFile.read(reinterpret_cast<char*>(graph), this->sizeofGraph());
    if (!inFile) {
      std::cout << "io error in KNN Graph file (tried to read graph) "
                << std::endl;
      inFile.close();
      return false;
    }

    inFile.read(reinterpret_cast<char*>(translation),
                this->sizeofTranslation());
    if (!inFile) {
      std::cout << "io error in KNN Graph file (tried to read translation) "
                << std::endl;
      inFile.close();
      return false;
    }

    inFile.read(reinterpret_cast<char*>(selection), this->sizeofSelection());
    if (!inFile) {
      std::cout << "io error in KNN Graph file (tried to read selection) "
                << std::endl;
      inFile.close();
      return false;
    }

    inFile.read(reinterpret_cast<char*>(nn1_stats), this->sizeofNN1Stats());
    if (!inFile) {
      std::cout << "io error in KNN Graph file (tried to read nn1_stats) "
                << std::endl;
      inFile.close();
      return false;
    }

    inFile.read(reinterpret_cast<char*>(nn1_dist_buffer),
                this->sizeofNN1DistBuffer());
    if (!inFile) {
      std::cout << "io error in KNN Graph file (tried to read nn1_dist_buffer) "
                << std::endl;
      inFile.close();
      return false;
    }

    std::cout << "read " << inFile.tellg() << "B of graph data " << std::endl;

    return true;
  }
};

template <typename KeyT, typename ValueT>
class KNNGraphWriter : private KNNGraphData<KeyT, ValueT> {
  std::ofstream outFile;
  bool configured;

 public:
  void configure(int N, int D, int K, int KF, int L, int S, float tau_build,
                 const char* description) {
    if (outFile.is_open()) {
      std::cout << "KNNGraphWriter::configure(): configure needs to be called "
                   "before opening the file."
                << std::endl;
      return;
    }
    this->header =
        KNNGraphHeader<KeyT, ValueT>(N, D, K, KF, L, S, tau_build, description);
    configured = true;
  }

  bool open(const std::string& filename) {
    if (!configured) {
      std::cout << "KNNGraphWriter::open(): please call configure() first"
                << std::endl;
      return false;
    }

    if (outFile.is_open()) {
      std::cout
          << "KNNGraphWriter::open(): A KNN Graph file was already opened."
          << std::endl;
      outFile.close();
    }

    outFile.open(filename, std::ofstream::out | std::ofstream::binary |
                               std::ofstream::trunc);
    if (!outFile.is_open()) {
      std::cout << "failed to open KNN Graph file for writing " << filename
                << std::endl;
      return false;
    }

    outFile.write(reinterpret_cast<const char*>(&this->header),
                  sizeof(this->header));
    if (!outFile) {
      std::cout << "io error in KNN Graph file (tried to write header) "
                << std::endl;
      outFile.close();
      return false;
    }

    std::cout << "opened KNN Graph file for writing " << filename << std::endl;

    return true;
  }

  void close() {
    if (outFile.is_open())
      outFile.close();
    else
      std::cout << "KNNGraphReader::close(): file was not open!" << std::endl;
  }

  bool write(const KeyT* graph, const KeyT* translation, const KeyT* selection,
             const ValueT* nn1_stats, const ValueT* nn1_dist_buffer) {
    if (!outFile.is_open()) {
      std::cout << "KNNGraphWriter::write(): file is not open!" << std::endl;
      return false;
    }

    outFile.write(reinterpret_cast<const char*>(graph), this->sizeofGraph());
    if (!outFile) {
      std::cout << "io error in KNN Graph file (tried to write graph) "
                << std::endl;
      outFile.close();
      return false;
    }

    outFile.write(reinterpret_cast<const char*>(translation),
                  this->sizeofTranslation());
    if (!outFile) {
      std::cout << "io error in KNN Graph file (tried to write translation) "
                << std::endl;
      outFile.close();
      return false;
    }

    outFile.write(reinterpret_cast<const char*>(selection),
                  this->sizeofSelection());
    if (!outFile) {
      std::cout << "io error in KNN Graph file (tried to write selection) "
                << std::endl;
      outFile.close();
      return false;
    }

    outFile.write(reinterpret_cast<const char*>(nn1_stats),
                  this->sizeofNN1Stats());
    if (!outFile) {
      std::cout << "io error in KNN Graph file (tried to write nn1_stats) "
                << std::endl;
      outFile.close();
      return false;
    }

    outFile.write(reinterpret_cast<const char*>(nn1_dist_buffer),
                  this->sizeofNN1DistBuffer());
    if (!outFile) {
      std::cout << "io error in KNN Graph file (tried to write nn1_dist_buffer) "
                << std::endl;
      outFile.close();
      return false;
    }

    std::cout << "wrote " << outFile.tellp() << "B of graph data " << std::endl;

    return true;
  }
};

#endif /* KNN_GRAPH_SERIALIZATION_HPP_ */
