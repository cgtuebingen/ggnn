FAQ
===

- Installing the ``ggnn`` Python module fails.

  Make sure you have the necessary :ref:`dependencies` installed
  and see the :ref:`troubleshooting` section.

- How do I benchmark datasets in HDF5 format, e.g. `ANN-Benchmarks`_?

  See the :ref:`ann-benchmarks-hdf5` section in the :doc:`benchmarking` page
  and the example file :file:`examples/python/sift1m_hdf5.py`
  for an example of how to load and process a HDF5 file with GGNN using Python.

.. _SIFT1B: http://corpus-texmex.irisa.fr/
.. _ANN-Benchmarks: https://github.com/erikbern/ann-benchmarks/

.. TODO: we should have an example Python script

Known Issues
------------

- Calling a GGNN function with invalid parameters crashes my program
  rather than returning an error / throwing an exception.

  GGNN was initially designed just to run benchmarks and is not yet fully transformed to a library.
  We are working on providing better user feedback in case of errors
  but sometimes the hard sanity checks will trigger and ``abort()`` the program.

  If there is any particular error case which should be handled better, please let us know.

- GGNN copies data through the CPU even though my multi-GPU setup supports peer to peer copies.

  Copying data from one GPU to another is not yet implemented.
  Typically, this only affects the query, which is quite small.
  When using a multi-GPU configuration, prefer to provide data on pinned CPU memory, whenever possible.

I can't find the answer to my question here
-------------------------------------------

Please feel free to open a new issue, we are happy to help!
