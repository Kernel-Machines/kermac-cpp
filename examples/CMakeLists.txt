
add_subdirectory(simple)

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
