project(kermac-example-cublas-simple-gemm)

add_executable(
    ${PROJECT_NAME}
    cublas_simple_gemm.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
    kermac-cublas
)
