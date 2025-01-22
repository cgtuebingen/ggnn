GGNN Documentation
==================

`GGNN`_  performs nearest-neighbor computations on CUDA-capable GPUs.
It supports billion-scale, high-dimensional datasets
and can execute on multiple GPUs through sharding.
When using just a single GPU, data can be exchanged directly with other code (e.g., torch tensors)
without copying through CPU memory.
GGNN is implemented using C++ and CUDA.
It can also be used from Python (>=3.8) via its `nanobind`_ bindings.

GGNN is based on the method proposed in the paper :ref:`GGNN: Graph-based GPU Nearest Neighbor Search <citing-this-project>`
by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, and Hendrik P.A. Lensch.
The original/official code corresponding to the published paper can be found in the `release_0.5`_ branch.

The :doc:`install` section explains how to install the library, and the :doc:`usage_python` and :doc:`usage_cpp` sections provides short tutorials and code examples.

.. _GGNN: https://github.com/cgtuebingen/ggnn
.. _release_0.5: https://github.com/cgtuebingen/ggnn/tree/release_0.5
.. _nanobind: https://github.com/wjakob/nanobind

Contents
--------

.. toctree::

  Home <self>
  install
  ann
  usage_python
  usage_cpp
  benchmarking
  FAQ

Capabilities and Limitations
----------------------------

The GGNN library supports...

- Billion-scale datasets with up to :math:`2^{31}-1` vectors.
- Data with up to 4096 dimensions.
- Building search graphs with up to 512 edges per node.
- Searching for up to 6000 nearest neighbors.
- Two distance measures: cosine and euclidean (L2) distance.

.. _citing-this-project:

Citing this Project
-------------------

You can use the following BibTeX entry to cite GGNN:

.. code-block:: bibtex

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


The official article can be found on IEEE, following this DOI: `10.1109/TBDATA.2022.3161156`_.
Alternatively, see the `ArXiV preprint`_.

.. _10.1109/TBDATA.2022.3161156: https://doi.org/10.1109/TBDATA.2022.3161156
.. _ArXiV preprint: https://arxiv.org/abs/1912.01059
