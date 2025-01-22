# GGNN: Graph-based GPU Nearest Neighbor Search

GGNN performs nearest-neighbor computations on CUDA-capable GPUs.
It supports billion-scale, high-dimensional datasets
and can execute on multiple GPUs through sharding.
When using just a single GPU, data can be exchanged directly with other code (e.g., torch tensors)
without copying through CPU memory.
GGNN is implemented using C++ and CUDA.
It can also be used from Python (>=3.8) via its [nanobind](https://github.com/wjakob/nanobind) bindings.

GGNN is based on the method proposed in the paper [GGNN: Graph-based GPU Nearest Neighbor Search](#citing-this-project)
by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, and Hendrik P.A. Lensch.
The original/official code corresponding to the published paper can be found in the [release_0.5](https://github.com/cgtuebingen/ggnn/tree/release_0.5) branch.

For more detailed information see our [documentation](https://ggnn.readthedocs.io/en/latest/).

## Installing the Python Module

### Prerequisites

GGNN is implemented in C++20/CUDA.
To compile and install the GGNN library, you need the [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) in version 12 or newer
and GCC/Clang C++ compilers version 10 or newer.

To run GGNN, a CUDA-capable GPU is required.

### Installing from PyPI

We're currently setting this up and will update the readme once this is ready.

### Manual Installation

Simply clone or download the repository and use pip to install the GGNN python module:

```bash
git clone https://github.com/cgtuebingen/ggnn.git
cd ggnn
python -m pip install .
```

## Compiling the C++/CUDA Code

To build the example programs for running benchmarks,
or to use GGNN with your own C++ or CUDA code, compile it using CMake:

```bash
git clone https://github.com/cgtuebingen/ggnn.git
cd ggnn
mkdir build
cd build
cmake ..
make -j4
```

### Prerequisites

- NVCC 12 or newer (CUDA Toolkit 12 or newer)
- either GCC (>=10) or Clang (>=10)
  (e.g., `g++-10` `libstdc++-10-dev` or `clang-10` `libc++-10-dev` `libc++abi-10-dev` on Ubuntu)
- `cmake` (>= 3.23)
- [nanobind](https://github.com/wjakob/nanobind)
  (`python -m pip install nanobind`)
- [glog](https://github.com/google/glog)
  (`libgoogle-glog-dev` on Ubuntu)
- [gflags](https://github.com/gflags/gflags)
  (`libgflags-dev` on Ubuntu)

The glog and gflags development libraries will be automatically fetched by CMake, if not installed.

### Troubleshooting

If your default C/C++ compilers are too old,
you may need to manually specify a newer version before running `cmake`:

```bash
export CC=gcc-10
export CXX=g++-10
export CUDAHOSTCXX=g++-10
```

For installation details, please see the [documentation](https://ggnn.readthedocs.io/en/latest/install.html).

## Example Usage

The GGNN python module can be used to perform GPU-accelerated approximate nearest-neighbor (ANN) queries using search graph or brute-force queries for determining the ground truth results.

* First, you need to setup a GGNN instance.
* Then, set the base dataset.
  * Datasets can be given as CPU/CUDA torch tensors or as numpy arrays.
* Given the base, you can build a search graph.
* Using the search graph, you can run queries.
* You can also run brute-force queries (no search graph required).
* The brute-force results can be used to evaluate the accuracy of the ANN queries.

```python
#! /usr/bin/python3

import pyggnn as ggnn
import torch

# get detailed logs
ggnn.set_log_level(4)


# create data
base = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')
query = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')


# initialize ggnn
my_ggnn = ggnn.GGNN()
my_ggnn.set_base(base)

# choose a distance measure
measure=ggnn.DistanceMeasure.Euclidean

# build the graph
my_ggnn.build(k_build=24, tau_build=0.5, refinement_iterations=2, measure=measure)


# run query
k_query: int = 10
tau_query: float = 0.64
max_iterations: int = 400

indices, dists = my_ggnn.query(query, k_query, tau_query, max_iterations, measure)


# run brute-force query to get a ground truth and evaluate the results of the query
gt_indices, gt_dists = my_ggnn.bf_query(query, k_gt=k_query, measure=measure)
evaluator = ggnn.Evaluator(base, query, gt_indices, k_query=k_query)
print(evaluator.evaluate_results(indices))

# print the indices of the 10 NN of the first five queries and their squared euclidean distances
print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')

```

For more examples in Python and in C++ see the [examples](https://github.com/cgtuebingen/ggnn/tree/release_0.9/examples) folder.
For more information about the parameters, on how to deal with data that is already on a GPU and on how to utilize multiple GPUs,
check out the [documentation](https://ggnn.readthedocs.io/en/latest/usage_python.html).
We also provide scripts that load and process typical [benchmark datasets](https://ggnn.readthedocs.io/en/latest/benchmarking.html).


## Capabilities and Limitations

The GGNN library supports...

- Billion-scale datasets with up to 2^31-1 vectors.
- Data with up to 4096 dimensions.
- Building search graphs with up to 512 edges per node.
- Searching for up to 6000 nearest neighbors.
- Two distance measures: cosine and euclidean (L2) distance.

## Citing this Project

You can use the following BibTeX entry to cite GGNN:

```bibtex
@ARTICLE{groh2022ggnn,
  author={Groh, Fabian and Ruppert, Lukas and Wieschollek, Patrick and Lensch, Hendrik P. A.},
  journal={IEEE Transactions on Big Data},
  title={GGNN: Graph-Based GPU Nearest Neighbor Search},
  year={2023},
  volume={9},
  number={1},
  pages={267-279},
  doi={10.1109/TBDATA.2022.3161156}
}
```

The official article can be found on IEEE, following this DOI: [10.1109/TBDATA.2022.3161156](https://doi.org/10.1109/TBDATA.2022.3161156).
Alternatively, see the [ArXiV preprint](https://arxiv.org/abs/1912.01059).

---
We hope this library makes your life easier and helps you to solve your problem!

Happy programming,
[Lukas Ruppert](https://github.com/LukasRuppert) and [Deborah Kornwolf](https://github.com/XDeboratti)

PS: If you have a question or a problem that occurs, please feel free to open an issue, we will be happy to help you.
