cmake_minimum_required(VERSION 3.12)
project(MatrixMultiplication C CXX)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)