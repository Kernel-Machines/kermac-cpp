project(kermac-example-cutensor-simple-contraction)

add_executable(
    ${PROJECT_NAME}
    cutensor_simple_contraction.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
    kermac-cutensor
)
