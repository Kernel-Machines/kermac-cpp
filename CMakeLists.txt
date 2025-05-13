cmake_minimum_required(VERSION 3.28)
project(kermac)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

add_subdirectory(src)
add_subdirectory(examples)
