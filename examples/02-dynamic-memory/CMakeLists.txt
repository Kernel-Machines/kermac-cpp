project(kermac-example-02-dynamic-memory)

add_executable(
    ${PROJECT_NAME}
    main.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
    kermac-cublas
)
