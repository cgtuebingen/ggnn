Installation
============

GGNN can be installed as a Python module or compiled as a library for C++/CUDA code.

.. _dependencies:

Dependencies
------------

The following dependencies are required to install the library:

- A C++20 compiler and standard library (GCC or Clang version 10 or higher)
- `CUDA Toolkit`_ version 12 or higher

  - This includes the Nvidia CUDA compiler ``nvcc``

The existence and version of these dependencies can be checked with::

   nvcc --version

and::

   c++ --version

Installing the GGNN Python Module
---------------------------------

To install GGNN, first the repository has to be cloned::

  git clone https://github.com/cgtuebingen/ggnn.git

The easiest way to install GGNN is from the folder containing the repository::

  cd ggnn

The `ggnn` module can then be installed using the package manager pip::

  python3 -m pip install .


.. caution::
  The PyPI package ``ggnn`` belongs to a different project.
  Running ``pip install ggnn`` will not install the GGNN Python module.


Installing the GGNN C++ Library
-------------------------------

To install GGNN, first the repository has to be cloned::

  git clone https://github.com/cgtuebingen/ggnn.git

The easiest way to install GGNN is from the folder containing the repository::

  cd ggnn

The GGNN library can then be built::

  mkdir build
  cd build
  cmake ..
  make -j4


.. _troubleshooting:

Troubleshooting
---------------

In case GGNN does not compile, check your CUDA and C++ compilers:

CUDA
  In case ``nvcc`` cannot be found by ``cmake``, you may get one of the following errors:

    - ``Failed to find nvcc.``
    - ``Compiler requires the CUDA toolkit.``
    - ``-- The CUDA compiler identification is unknown``
    - ``Failed to detect a default CUDA architecture.``


  Set the ``PATH`` and ``LD_LIBRARY_PATH`` to your installed `CUDA Toolkit`_, e.g.:

    .. code-block:: bash

      export PATH="/usr/local/cuda-12.4/bin/:${PATH}"
      export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64/:${LD_LIBRARY_PATH}"

  Now, ``nvcc --version`` should print something like this::

      nvcc: NVIDIA (R) Cuda compiler driver
      Copyright (c) 2005-2024 NVIDIA Corporation
      Built on Thu_Mar_28_02:18:24_PDT_2024
      Cuda compilation tools, release 12.4, V12.4.131
      Build cuda_12.4.r12.4/compiler.34097967_0

C++
  For compilation, we support GCC's ``g++`` >= 10 and LLVM's ``clang++`` >= 10.

  If you use a different compiler or an outdated version, you may get the following message:

    - ``GCC or Clang version 10 or higher required for C++20 support!``

  You can define which C/C++ compilers should be used with the following environment variables:

    .. code-block:: bash

      # E.g., to use GCC 10, set the following:
      export CC=gcc-10
      export CXX=g++-10
      export CUDAHOSTCXX=g++-10

  Also, make sure to have the matching C++ standard library version installed.

    For GCC 10, install the following on Ubuntu:

      ``g++-10`` and ``libstdc++-10-dev``

    For Clang 10, install the following on Ubuntu:

      ``clang-10``, ``libc++-10-dev``, and ``libc++abi-10-dev``

    Similarly for newer versions.

  This has been tested on Ubuntu 20.04.
  Newer versions will ship newer versions by default.
  E.g., GCC 13 and Clang 18 on Ubuntu 24.04,
  which should work out-of-the-box.

CMake
  Make sure to re-run ``cmake`` in a fresh ``build`` folder after exporting these environment variables.
  Otherwise, ``cmake`` may use settings from a cached configuration.


.. _CUDA Toolkit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
