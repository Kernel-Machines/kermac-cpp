cmake_minimum_required(VERSION 3.28)
project(kermac)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

option(KERMAC_BASE_ONLY "Enable kermac component only" OFF)
option(KERMAC_CUBLAS "Enable kermac-cublas component" ON)
option(KERMAC_CUSOLVER "Enable kermac-cusolver component" ON)
option(KERMAC_CUTENSOR "Enable kermac-cutensor component" ON)

add_subdirectory(kermac)

if (NOT KERMAC_BASE_ONLY)
    if(KERMAC_CUBLAS)
        add_subdirectory(kermac-cublas)
    endif()
    if(KERMAC_CUSOLVER)
        add_subdirectory(kermac-cusolver)
    endif()
    if(KERMAC_CUTENSOR)
        add_subdirectory(kermac-cutensor)
    endif()
endif()


add_subdirectory(examples)
