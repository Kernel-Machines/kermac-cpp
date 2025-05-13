project(kermac-cublas)

add_library(kermac-cublas INTERFACE)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    .
    include
)

target_link_libraries(
    ${PROJECT_NAME} 
    INTERFACE 
    kermac 
    cublas
)