Using the GGNN C++ Library
==========================

This section explains how to use the GGNN C++ library.

You can find all the code from this tutorial and additional example files in the :file:`examples/cpp-and-cuda/` folder of the GGNN repository.

Including GGNN
--------------

Before using GGNN, the ``ggnn/base/ggnn.cuh`` header has to be included from the GGNN library.
For convenience we include some parts of the standard library
and we use the ``ggnn`` namespace to avoid prefixing all GGNN classes with ``ggnn::``.

.. code:: c++

  #include <ggnn/base/ggnn.cuh>
  #include <array>
  #include <iostream>
  #include <cstdint>
  #include <random>

  using namespace ggnn;

The header files from the standard library are only for demonstration purposes and are not required for using the library.

Using CPU Data
--------------

Then, some data to search in and some data to search the *k* nearest neighbors for is needed.
Instead of a ``std:array`` you can also use a ``std::vector``
or any other standard C++ container which can be mapped to a ``std::span``:

.. code:: c++

  int main() {

    const size_t N_base = 10'000;
    const size_t N_query = 10'000;
    const uint32_t dim = 128;

    // the data to query on
    std::array<float, N_base*dim> base_data;
    // the data to query for
    std::array<float, N_query*dim> query_data;

    // setup a random number generator
    std::default_random_engine prng {};
    std::uniform_real_distribution<float> uniform{0.0f, 1.0f};

    // generate the random data
    for(float& x : base_data)
      x = uniform(prng);
    for (float& x : query_data)
      x = uniform(prng);

Then, a GGNN instance and the datasets can be initialized:

.. code:: c++

    /// data type for addressing points
    using KeyT = int32_t;
    /// data type of computed distances
    using ValueT = float;
    using GGNN = GGNN<KeyT, ValueT>;

    // Initialize GGNN
    GGNN ggnn{};

    // Initialize the datasets containing the base data and query data
    Dataset<float> base = Dataset<float>::copy(base_data, dim, true);
    Dataset<float> query = Dataset<float>::copy(query_data, dim, true);

Instead of copying the data, data on the host can also be referenced with ``referenceCPUData()`` and data on the GPU can be referenced with ``referenceGPUData()``.

.. caution::

  When referencing data, make sure its lifetime exceeds the lifetime of the GGNN instance.

If the data is a dataset in fvecs or bvecs format it can be loaded with ``Dataset<BaseT>::load(path_to_file)``.

The base has to be passed to GGNN:

.. code:: c++

    ggnn.setBaseReference(base);

Now, GGNN is ready to be used and a graph can be built:

.. code:: c++

    // build the search graph
    ggnn.build(/*k_build*/ 24, /*tau_build*/ 0.5f);

The parameters are the same as when :doc:`usage_python` and are also further explained in the :ref:`search graph parameters` section.
In addition to ``k_build`` and ``tau_build``, you can also specify the number of ``refinement_iterations`` and the ``measure``.
The measure can either be ``DistanceMeasure::Euclidean`` or ``DistanceMeasure::Cosine``.

.. code:: c++

    // run the query and store indices & squared distances
    const uint32_t KQuery = 10;
    const auto [indices, dists] = ggnn.query(query, KQuery, /*tau_query*/ 0.5f);

The parameters of the query are again the same as when :doc:`usage_python` and further explained in the :ref:`query parameters` section.
You can specify the ``query``, ``KQuery``, ``tau_query``, ``max_iterations``, and the ``measure``.

Finally, the example program prints the indices and squared euclidean distances of the 10 nearest neighbors of the first query:

.. code:: c++

    // print the results for the first query
    std::cout << "Result for the first query vector: \n";
    for(uint32_t i=0; i < KQuery; i++){
        //std::cout << "Base Idx: ";
        std::cout << "Distance to vector at base[";
        std::cout.width(5);
        std::cout << indices[i];
        std::cout << "]: " << dists[i] << "\n";
    }
    return 0;
  }


Using GPU Data
--------------

In the following, the data is assumed to already be located on the GPU.
For demonstration purposes, we generate some random data using `cuRAND`_:

.. code:: c++

  #include <ggnn/base/ggnn.cuh>
  #include <ggnn/base/eval.h>

  #include <cstdint>
  #include <iostream>

  #include <cuda_runtime.h>
  #include <curand.h>

  using namespace ggnn;

  int main() {

    /// data type for addressing points
    using KeyT = int32_t;
    /// data type of computed distances
    using ValueT = float;
    using GGNN = GGNN<KeyT, ValueT>;

    //create data on gpu
    size_t N_base {10'000};
    size_t N_query {10'000};
    uint32_t D {128};

    float* base;
    float* query;

    // allocate GPU data
    cudaMalloc(&base, N_base*D*sizeof(float));
    cudaMalloc(&query, N_query*D*sizeof(float));

    // setup the random number generator
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

    // generate some random data
    curandGenerateUniform(generator, base, N_base*D);
    curandGenerateUniform(generator, query, N_query*D);

Next, GGNN has to be initialized and, to avoid a copy, the data can be referenced:

.. code:: c++

  // Initialize GGNN
  GGNN ggnn{};

  // Set the data on the GPU as the base dataset on which the graph should be built on.
  // To reference existing data, specify its pointer, the number of base vectors N_base,
  // the dimensionality of base vectors D and the gpu_id of the GPU containing the data.
  int32_t gpu_id = 0;
  ggnn.setBase(ggnn::Dataset<float>::referenceGPUData(base, N_base, D, gpu_id));

  // Also reference the query data already existing on the GPU
  auto d_query = ggnn::Dataset<float>::referenceGPUData(query, N_query, D, gpu_id);

Now, build a search graph using GGNN and run a query:

.. code:: c++

    // build the search graph
    const uint32_t KBuild = 24;
    const float tau_build = 0.5f;
    ggnn.build(KBuild, tau_build);

    // run the query and store indices & distances
    const int32_t KQuery = 10;
    const auto [indices, dists] = ggnn.query(d_query, KQuery, 0.5);

    // print the results for the first query
    std::cout << "Result for the first query verctor: \n";
    for(uint32_t i=0; i < KQuery; i++){
      //std::cout << "Base Idx: ";
      std::cout << "Distance to vector at base[";
      std::cout.width(5);
      std::cout << indices[i];
      std::cout << "]: " << dists[i] << "\n";
    }

.. note::

  While the query data is given on the GPU, results are still returned to the CPU by default.

Finally, some cleanup.

.. code:: c++

    // cleanup
    curandDestroyGenerator(generator);
    cudaFree(base);
    cudaFree(query);

    return 0;
  }

.. _cuRAND: https://docs.nvidia.com/cuda/curand/index.html

Using multiple GPUs
-------------------

To work on multiple GPUs, GGNN uses sharding.

A shard is a portion of the base dataset, for which an individual search graph "graph shard" is built.
To make sure no base vector is left out, the base dataset needs to be evenly divisible by ``shard_size``.
During query, all graph shards are being searched and the results of all shards are then merged on the CPU.
Shards are equally distributed across all GPUs.
Therefore, the number of shards has to be evenly divisible by the number of GPUs used.

To tell GGNN which GPUs to use, use the ``setGPUs`` method.
To set the shard size, use ``setShardSize``:

.. code:: c++

  // initialize GGNN
  GGNN ggnn;

  ggnn.setBaseReference(base);

  // configure which GPUs to use
  ggnn.setGPUs({0,1});

  // split dataset into shards of this size
  ggnn.setShardSize(25'000);

In case the GPU memory is insufficient to keep all assigned graph and base shards in memory,
shards will automatically be swapped out to CPU memory and to disk.
You can specify a CPU memory limit and the directory in which the swapped out shards will be stored.

.. code:: c++

  // use 64 GB of CPU memory for swapping out shards
  const size_t available_memory = 64UL * 1024 * 1024 * 1024;
  ggnn.setCPUMemoryLimit(available_memory);

  ggnn.setWorkingDirectory("/some/path/for/swapping/out/shards");

Once everything is setup, build and query the search graph as usual:

.. code:: c++

  // build a search graph for all shards
  ggnn.build(/*KBuild*/ 24, /*tau_build*/ 0.5f);

  // query all shards and return the merged result
  const auto [indices, dists] = ggnn.query(query, KQuery, /*tau_query*/ 0.5f);


Loading Datasets (e.g. SIFT1M)
------------------------------

GGNN can load datasets in ``.fvecs``, ``.bvecs``, and ``.ivecs`` format
for benchmark datasets such as `SIFT1M`_ and `SIFT1B`_.

.. code:: c++

  Dataset<float> sift1m_base = Dataset<float>::load("/path/to/sift_base.fvecs");
  Dataset<unsigned char> sift1b_base = Dataset<unsigned char>::load("/path/to/bigann_base.bvecs");
  Dataset<int> sift1m_gt = Dataset<int>::load("/path/to/sift_groundtruth.ivecs");


.. _SIFT1M: http://corpus-texmex.irisa.fr/
.. _SIFT1B: http://corpus-texmex.irisa.fr/

