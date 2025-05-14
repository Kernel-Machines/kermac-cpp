
add_subdirectory(01-simple)
add_subdirectory(02-dynamic-memory)

if (KERMAC_CUBLAS)
    add_subdirectory(cublas-simple-gemm)
else()
    message(STATUS "Not building kermac-cublas examples")
endif()

if (KERMAC_CUSOLVER)
    add_subdirectory(cusolver-simple-syevd)
else()
    message(STATUS "Not building kermac-cusolver examples")
endif()

if (KERMAC_CUTENSOR)
    add_subdirectory(cutensor-simple-contraction)
else()
    message(STATUS "Not building kermac-cutensor examples")
endif()