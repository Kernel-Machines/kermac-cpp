project(kermac-cusolver)

add_library(kermac-cusolver INTERFACE)

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
    cusolver
)