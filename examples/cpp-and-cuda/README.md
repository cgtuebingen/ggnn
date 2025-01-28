# GGNN Example Code

The files in this folder are compiled as part of the main GGNN CMake project.

## CMake Example

Additionally, this folder contains a `CMakeLists.txt`
describing an example CMake project for using GGNN as an installed library.

First, you need to compile and install GGNN using CMake:

```bash
# in the main GGNN repository
mkdir build
cd build
# by default, CMake would install system-wide
# by specifying the CMAKE_INSTALL_PREFIX,
# you can install GGNN to your home instead
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
make -j4
make install
```

Then, you can use the GGNN library within your own CMake projects.

As a minimal example, see the `CMakeLists.txt` in this folder.
