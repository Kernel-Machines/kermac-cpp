project(kermac-example-cublas-simple-gemm)

add_executable(
    ${PROJECT_NAME}
    main.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
    kermac-cublas
)
