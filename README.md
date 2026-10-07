# Parallel Matrix Multiplication

A C++ project demonstrating the performance improvements of matrix multiplication using OpenMP. The program compares the execution time of a single-threaded implementation against an OpenMP parallelized version and calculates the total speedup.

## Prerequisites

- C++23 compatible compiler
- CMake 3.12 or higher
- OpenMP

## Build Instructions

To build the project, run the following commands from the root directory:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Run Instructions

After a successful build, you can run the executable from the `build` directory:

```bash
./main
```

## Description

The program executes the following steps:
1. Initializes two matrices (`A` and `B`) with random integers.
2. Computes the product `C = A * B` on a single thread over multiple samples, recording the average time.
3. Computes the product using OpenMP parallelization over the same number of samples, recording the average time.
4. Outputs the average time for both methods and the resulting performance speedup.
