project(kermac-example-01-simple)

add_executable(
    ${PROJECT_NAME}
    main.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    kermac
)
