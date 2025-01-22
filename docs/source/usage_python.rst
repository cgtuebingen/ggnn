Using the GGNN Python Module
============================

This section explains how to use the ``ggnn`` Python module.

While written in C++/CUDA, GGNN can be used from Python via its `nanobind`_ bindings.

.. _nanobind: https://github.com/wjakob/nanobind


The code from this tutorial and additional examples can be found in the :file:`ggnn/examples/python/ggnn_pytorch.py` file of the GGNN repository.

Importing GGNN
--------------

After installing the ggnn module, it needs to be imported.
``ggnn.set_log_level(4)`` enables verbose logging of information into the console during the execution of the algorithm.
The higher the log level (``0`` to ``4``), the more information is printed.
By default, the log level is set to ``0``:

.. code:: python

  #! /usr/bin/python3

  import ggnn

  #get detailed logs
  ggnn.set_log_level(4)

Using CPU Data
--------------

For demonstration purposes, we will create some random example data using torch:

.. code:: python

  import torch

  #create data
  base = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')
  query = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')

.. note::

  You can also use numpy arrays instead of torch tensors.


The next step is to create an instance of the GGNN class from the ggnn module. The GGNN class needs the base data (``my_ggnn.set_base(base)``) and can then build the graph:

.. code:: python

  # initialize ggnn
  my_ggnn = ggnn.GGNN()
  my_ggnn.set_base(base)

  # choose a distance measure
  measure=ggnn.DistanceMeasure.Euclidean

  # build the graph
  my_ggnn.build(k_build=24, tau_build=0.5, refinement_iterations=2, measure=measure)


The parameters of the ``build(k_build, tau_build, measure)`` function need some explanation.
``k_build >= 2`` describes the number of outgoing edges per node in the graph.
The larger ``k_build``, the longer the build time and the query.
``tau_build`` influences the stopping criterion during the creation of the graph.
The larger the ``tau_build``, the longer the build time.
Typically, :math:`0.3 < \tau_{build} < 1` is enough to get good results during search.
It is recommended to experiment with these parameters to get the best possible trade-off between build time and accuracy out of the search.
See the paper :ref:`GGNN: Graph-based GPU Nearest Neighbor Search <citing-this-project>` and the :ref:`search graph parameters` section for more information on parameters and some examples.
``measure`` is the distance measure to compare the distances of the vectors.
The ggnn module supports cosine and euclidean (L2) distance, euclidean distance is the default, so passing this parameter is optional.

.. caution::

  The distance measure for building, querying and computing the ground truth should be the same.
  If set explicitly, make sure to provide its value to all functions.


Now, the approximate nearest neighbor search can be performed:

.. code:: python

  # run query
  k_query: int = 10
  tau_query: float = 0.64
  max_iterations: int = 400

  indices, dists = my_ggnn.query(query, k_query, tau_query, max_iterations, measure)


The parameters of the ``query(query, k_query, tau_query, max_iterations, measure)`` are:

- ``query`` are all the vectors, to search the *k* nearest neighbors for.
- ``k_query`` tells the search algorithm how many neighbors it should return per query vector.
  Generally, the higher ``k_query``, the longer the search.
  The ggnn module supports up to 6000 neighbors, but it is recommended to search only for 10-1000 neighbors.
- ``tau_query`` and ``max_iterations`` determine the stopping criterion.
  For both parameters it holds that the larger the parameter, the longer the search.
  Typically, :math:`0.7 < \tau_{query} < 2` and :math:`200 < max\_iterations < 2000` is enough to get good results during search.
- ``measure`` is the distance measure that is used to compute the distances between vectors. ``Euclidean`` is the default, so this parameter is optional. To set cosine similarity you can pass ``measure=ggnn.DistanceMeasure.Cosine`` as parameter.


In this example, a ground truth is computed via a brute-force query and the result of the ANN search is evaluated:

.. code:: python

  # run brute-force query to get a ground truth and evaluate the results of the query
  gt_indices, gt_dists = my_ggnn.bf_query(query, k_gt=k_query, measure=measure)
  evaluator = ggnn.Evaluator(base, query, gt_indices, k_query=k_query)
  print(evaluator.evaluate_results(indices))

For computing a ground truth, we need  to pass ``k_gt`` which should be at least as many as ``k_query`` if we want to compare properly.
In case of duplicates in the dataset, a larger set of ground truth indices can be used to accurately determine the accuracy.

.. note::

  The brute-force query can only be run in single-GPU mode.


After evaluating the example program prints the indices of the *k* nearest neighbors for the first five queries and their squared euclidean distances:

.. code:: python

  # print the indices of the 10 NN of the first five queries and their squared euclidean distances
  print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')


Using GPU Data
--------------

This works just like with data on the host, but the device of the torch tensors must be set to ``device='cuda'``
and possibly the respective GPU index must be added, e.g. ``device='cuda:1'``.

GGNN can return the result of the *k* nearest neighbor search on the GPU with ``my_ggnn.set_return_results_on_gpu(True)``.
If not set, the results will be on the host.

.. note::

  Returning the results on the GPU is not possible in a multi-GPU setup.
  When using sharding, sorted results of all shards are returned (since merging would be performed on the CPU).


.. code:: python

  #create data
  base = torch.rand((10_000, 128), dtype=torch.float32, device='cuda')
  query = torch.rand((10_000, 128), dtype=torch.float32, device='cuda')

  # initialize GGNN
  my_ggnn = ggnn.GGNN()
  my_ggnn.set_base(base)
  my_ggnn.set_return_results_on_gpu(True)



Using Multiple GPUs
-------------------

To work on multiple GPUs, GGNN uses sharding.

A shard is a portion of the base dataset, for which an individual search graph "graph shard" is built.
To make sure no base vector is left out, the base dataset needs to be evenly divisible by ``shard_size``.
During query, all graph shards are being searched and the results of all shards are then merged on the CPU.
Shards are equally distributed across all GPUs.
Therefore, the number of shards has to be evenly divisible by the number of GPUs used.

To tell the ggnn instance which GPUs to use, use the ``set_gpus(gpu_ids)`` function, which expects a list of CUDA device ids.
To set the shard size, use ``set_shard_size(n_shard)``, where ``n_shard`` describes the number of base vectors that should be processed at once.

Otherwise, this works the same way as above.

.. code:: python

  #! /usr/bin/python3

  import ggnn
  import torch

  # create data
  base = torch.rand((100_000, 128), dtype=torch.float32, device='cpu')
  query = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')

  # initialize ggnn and prepare multi-GPU
  my_ggnn = ggnn.GGNN()
  my_ggnn.set_base(base)
  my_ggnn.set_shard_size(n_shard=25_000)
  my_ggnn.set_gpus(gpu_ids=[0,1])

  # build the graph
  my_ggnn.build(k_build=24, tau_build=0.9)

  # run query
  indices, dists = my_ggnn.query(query, k_query=10, tau_query=0.64, max_iterations=400)

  print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')

.. caution::

  Copying data between different GPUs is not supported.
  Instead, data is automatically copied from GPU to CPU and then to a different GPU, if necessary.
  When using multiple GPUs, it should therefore be preferred to provide the data on the CPU.

  Since query results from multiple GPUs are merged on the CPU,
  returning them on the GPU is also not possible.

.. tip::

  To achieve a multi-GPU, GPU-only setup,
  setup multiple independent instances of GGNN, one per GPU,
  and merge the query results yourself.


Loading Datasets (e.g. SIFT1M)
------------------------------

If the data is provided in :file:`.fvecs` or  :file:`.bvecs` format, as for example the `SIFT1M`_ and `SIFT1B`_ datasets,
the dataset can be loaded using the ``.load('/path/to/file')`` function.
Besides a ``FloatDataset`` (``float``), the ggnn module can also load a base and query as ``UCharDataset`` (``unsigned char``).
If a ground truth is provided as an :file:`.ivecs` file, it can be loaded as an ``IntDataset`` (``int``)
and passed to the ``Evaluator`` directly.

.. code:: python

  #! /usr/bin/python3

  import ggnn

  path_to_dataset = '/path/to/sift/'

  base = ggnn.FloatDataset.load(path_to_dataset + 'sift_base.fvecs')
  query = ggnn.FloatDataset.load(path_to_dataset + 'sift_query.fvecs')
  gt = ggnn.IntDataset.load(path_to_dataset + 'sift_groundtruth.ivecs')

  k_query: int = 10

  evaluator = ggnn.Evaluator(base, query, gt, k_query)

  my_ggnn = ggnn.GGNN()
  my_ggnn.set_base(base)
  my_ggnn.build(k_build=24, tau_build=0.5)

  indices, dists = my_ggnn.query(query, k_query, tau_query=0.64, max_iterations=400)
  print(evaluator.evaluate_results(indices))


.. _SIFT1M: http://corpus-texmex.irisa.fr/
.. _SIFT1B: http://corpus-texmex.irisa.fr/

