project(kermac-example-simple)

add_executable(
    ${PROJECT_NAME}
    simple.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
)
