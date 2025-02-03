Benchmarking
============

Running Standardized Benchmarks
-------------------------------

In order to run standardized ANN benchmarks, you can use the Python module
to run your own benchmark scripts
or you can run the example program :program:`ggnn_benchmark`
which is compiled alongside the C++ library
and can be applied to arbitrary datasets.

Everything dataset-specific can be configured via the following command line parameters:

``base``
  Path to the base dataset ``.fvecs`` or ``.bvecs`` file.

``subset`` (optional)
  In case you want to only load a subset of the base dataset,
  you can specify the size of that subset here.
  Only the first ``subset`` many points will be loaded.
  By default, or if set to ``0``, the entire base dataset file will be loaded.

``query``
  Path to the query dataset ``.fvecs`` or ``.bvecs`` file.

``gt`` (optional)
  Path to the ground truth indices ``.ivecs`` file.

  .. note::

    If not given, the ground truth will be brute-forced, if possible.

    If a file name is given, but the file does not exist, the brute-forced result will be stored.

``graph_dir`` (optional)
  Directory for loading/storing the GGNN graph or graph shards.


  .. note::

    If the directory already contains a GGNN graph, it will be loaded and construction will be skipped.
    Otherwise, the constructed graph will be stored in this directory.

  .. note::
    If left empty, the graph will be discarded when the program ends.

    If necessary (i.e., if GPU memory is insufficient to keep all shards loaded),
    GGNN will still swap out shards from GPUs to RAM and disk automatically in multi-shard settings.

    In that case, when no directory is specified, GGNN graph shards will be stored in the current working directory.

``k_build`` (optional, default ``24``)
  Number of neighbors per point in the search graph (see :ref:`search graph parameters`).

``tau_build`` (optional, default ``0.5``)
  Slack factor for search graph construction (see :ref:`search graph parameters`).

``refinement_iterations`` (optional, default: ``2``)
  Number of iterations for search graph refinement.

``k_query`` (optional, default ``10``)
  Number of neighbors to search for (see :ref:`query parameters`).

``max_iterations`` (optional, default ``200``)
  Maximum number of query iterations per search (see :ref:`query parameters`).

``measure`` (optional, default ``euclidean``)
  Distance measure (``euclidean`` or ``cosine``) (see :ref:`distance measures`).

``shard_size`` (optional)
  Number of points per shard.
  With sharding, the base datasets is split into equally-sized shards.
  This parameter defines the size of one shard.

  .. caution::

    The base dataset needs to be evenly divisible by the shard size.
    The resulting number of shards needs to be evenly divisible by the number of GPUs.

``gpu_ids`` (optional)
  CUDA device indices of the GPUs to be used by GGNN, separated by spaces.
  E.g., ``'0 1 2 3'``.

  .. note::

    Using multiple GPUs requires sharding (see ``shard_size``).

  .. tip::

    CUDA device indices can be influenced by the `CUDA Environment Variables`_
    ``CUDA_VISIBLE_DEVICES`` and ``CUDA_DEVICE_ORDER``.

``grid_search`` (optional)
  If set, run a larger sweep of queries with :math:`\tau_{query} \in [0.7, 2.0]`
  rather than just a small set of queries.

``v`` (optional)
  Verbosity level between ``0`` and ``4`` (maximum verbosity).


.. _CUDA Environment Variables: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars


.. code:: bash

  ./build/ggnn_benchmark \
    --base /path/to/sift_base.fvecs \
    --query /path/to/sift_query.fvecs \
    --gt /path/to/sift_groundtruth.ivecs \
    --graph_dir ./ \
    --tau_build 0.5 \
    --refinement_iterations 2 \
    --k_build 24 \
    --k_query 10 \
    --measure euclidean \
    --shard_size 0 \
    --subset 0 \
    --gpu_ids 0 \
    --grid_search false


.. _ann-benchmarks-hdf5:

ANN-Benchmarks / HDF5
---------------------

In order to run a benchmark from `ANN-Benchmarks`_, you might want to load a dataset from an HDF5 file.
You can do so with a simple Python script:

.. code:: python

  import h5py
  import numpy as np

  # load ANN-benchmark-style HDF5 dataset
  with h5py.File(path_to_dataset, 'r') as f:
    base = np.array(f['train'])
    query = np.array(f['test'])
    gt = np.array(f['neighbors'])


See also the example file :file:`examples/python/sift1m_hdf5.py`.

.. _ANN-Benchmarks: https://github.com/erikbern/ann-benchmarks/


Reference Configurations
------------------------

The default values set in the :program:`ggnn_benchmark` program are set for the `SIFT1M`_ dataset.
For other datasets, set the parameters as documented in the GGNN paper.

.. TODO: parameters per dataset and some expected query results.

.. note::
  We will update this documentation shortly to reference all necessary configurations.

  For now, check the ``.cu`` files per dataset under ``src`` in the `release_0.5`_ branch
  and the official paper :ref:`GGNN: Graph-based GPU Nearest Neighbor Search <citing-this-project>`.


.. _SIFT1M: http://corpus-texmex.irisa.fr/
.. _release_0.5: https://github.com/cgtuebingen/ggnn/tree/release_0.5

