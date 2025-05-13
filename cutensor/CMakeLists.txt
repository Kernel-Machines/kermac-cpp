cmake_minimum_required(VERSION 3.22)

project(cutensor)

# Option to control cuTENSOR linking
option(USE_CUTENSOR_STANDALONE "Enable cuTENSOR downloading and linking" ON)

string(REGEX MATCH "^[0-9]+" CUDA_MAJOR_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

if (CUDA_MAJOR_VERSION EQUAL 12)
    set(CUDA_VERSION "12")        # Match your CUDA version
elseif (CUDA_MAJOR_VERSION EQUAL 11)
    set(CUDA_VERSION "11")
else()
    message(FATAL_ERROR "Unsupported CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}, requires 11 or 12")
endif()

# Specify cuTENSOR version and platform
set(CUTENSOR_VERSION_MAJOR "2")
set(CUTENSOR_VERSION_SHORT "2.2.0")
set(CUTENSOR_VERSION "2.2.0.0")  # Update to desired version

set(OS "linux")           # Adjust for your OS (e.g., ubuntu20.04, rhel8)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(ARCH "sbsa")            # Adjust if needed (e.g., aarch64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ARCH "x86_64")
else()
    message(FATAL_ERROR "Unsupported architecture detected: ${CMAKE_SYSTEM_PROCESSOR}. This project only supports ARM (arm/aarch64) or x86_64 (x86_64/amd64).")
endif()

file(MAKE_DIRECTORY ${DOWNLOADS_DIR})
# Construct the URL (based on NVIDIA's typical pattern)
set(TAR_URL "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/${OS}-${ARCH}/libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}-archive.tar.xz")
set(TAR_FILE "${DOWNLOADS_DIR}/libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}-archive.tar.xz")
set(EXTRACT_DIR "${DOWNLOADS_DIR}")
set(CUTENSOR_ROOT "${EXTRACT_DIR}/libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}-archive")

if(NOT EXISTS "${TAR_FILE}")
    message(STATUS "Downloading cutensor ${CUTENSOR_VERSION}")
    # Download the tar file
    file(DOWNLOAD 
        "${TAR_URL}" 
        "${TAR_FILE}"
        SHOW_PROGRESS
        STATUS download_status
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download cuTENSOR: ${download_status}")
    endif()
else()
    message(STATUS "Using existing cuTENSOR tar file: ${TAR_FILE}")
endif()

if(NOT EXISTS "${CUTENSOR_ROOT}")
    set(DESIRED_FILES 
        "libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}-archive/include"
        "libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}-archive/lib/${CUDA_VERSION}/libcutensor.so.${CUTENSOR_VERSION_SHORT}"
        "libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}-archive/lib/${CUDA_VERSION}/libcutensor.so.${CUTENSOR_VERSION_MAJOR}"
        "libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}-archive/lib/${CUDA_VERSION}/libcutensor.so"
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xJf "${TAR_FILE}" ${DESIRED_FILES}
        WORKING_DIRECTORY "${EXTRACT_DIR}"
        RESULT_VARIABLE extract_result
    )

    if(NOT extract_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract cuTENSOR")
    endif()
else()
    message(STATUS "Using existing extracted cuTENSOR: ${EXTRACT_DIR}")
endif()

if (NOT USE_CUTENSOR_STANDALONE)
# Set up library paths for your project

set(CUTENSOR_INCLUDE "${CUTENSOR_ROOT}/include")
set(CUTENSOR_LIB "${CUTENSOR_ROOT}/lib/${CUDA_VERSION}")

add_library(${PROJECT_NAME} INTERFACE)

# Add to your project
target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    ${CUTENSOR_INCLUDE}
)

target_link_libraries(
    ${PROJECT_NAME}
    INTERFACE
    ${CUTENSOR_LIB}/libcutensor.so
)
endif()
