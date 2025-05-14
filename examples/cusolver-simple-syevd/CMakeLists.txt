project(kermac-example-cusolver-simple-syevd)

add_executable(
    ${PROJECT_NAME}
    main.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
    kermac-cublas
    kermac-cusolver
)
