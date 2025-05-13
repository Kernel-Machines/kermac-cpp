project(kermac-cutensor)

add_library(kermac-cutensor INTERFACE)

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
    cutensor
)
