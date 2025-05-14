project(kermac-example-cutensor-simple-contraction)

add_executable(
    ${PROJECT_NAME}
    main.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
    kermac-cutensor
)
