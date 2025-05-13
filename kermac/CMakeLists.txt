add_library(kermac INTERFACE)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    .
    include
)

target_link_libraries(${PROJECT_NAME} INTERFACE cublas)