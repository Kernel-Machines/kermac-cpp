#pragma once

#include <inttypes.h>
#include <cassert>
#include <stdio.h>

#include <cublas_v2.h>

using i32 = int32_t;
using i64 = int64_t;

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

template <typename T> constexpr T c_one = T(1.0);
template <typename T> constexpr T c_zero = T(0.0);
template <typename T> constexpr T c_two = T(2.0);
template <typename T> constexpr T c_negative_one = T(-1.0);
template <typename T> constexpr T c_negative_two = T(-2.0);

#define ASSERT(X) \
	do {                                                            		\
		if (!(X)) {                                                 		\
			assert(false);                                          		\
			printf("Error in file %s at line %d\n", __FILE__, __LINE__);	\
			exit(EXIT_FAILURE);                                     		\
		}                                                           		\
	} while(0)

#define cuCheck(ans) gpuAssert((ans), __FILE__, __LINE__, 1)
#define cuCheck_no_abort(ans) gpuAssert((ans), __FILE__, __LINE__, 0)
static void gpuAssert(cudaError_t error, const char *file, int line, int abort) {
	if (error != cudaSuccess) {
			const char* error_name = cudaGetErrorName(error);
			const char* error_string = cudaGetErrorString(error);

			fprintf(stderr, "GPUassert: %s \n  %s \n  %s %d\n", error_name, error_string, file, line);
		if (abort) assert(0);
	}
}

#define cublasCheck(ans) checkCublasStatus((ans), __FILE__, __LINE__)
static void checkCublasStatus(cublasStatus_t status, const char *file, int line) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS API failed with status %d %s %d\n", status, file, line);
		exit(1);
	}
}

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

#define NEAREST_LARGER_MULTIPLE(X,Y) ((((X) - 1) / Y) + 1)
#define NEAREST_SMALLER_MULTIPLE(X,Y) ((X) / (Y))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ARRAY_NUM_ELEMS(array) sizeof((array)) / sizeof(*(array))

#define PRINT_DRY(a, fmt, ...) do { if (a.is_dry_run()) { printf((fmt), ##__VA_ARGS__); } } while (0)
#define PRINT_WET(a, fmt, ...) do { if (!a.is_dry_run()) { printf((fmt), ##__VA_ARGS__); } } while (0)

#define DRY_RUN(a) a.is_dry_run()
#define DRY_RETURN(a) do { if (a.is_dry_run()) return; } while (0)
#define DRY_COND(a) if (a.is_dry_run())
#define WET_COND(a) if (!a.is_dry_run())

#define STACK_EXECUTE(dsa, hsa, block)  \
    do {                                \
        block;                          \
        (dsa).alloc();                  \
        (hsa).alloc();                  \
        block;                          \
        (dsa).clear();                  \
        (hsa).clear();                  \
    } while (0)

#include <chrono>

struct CPUTimer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    
    CPUTimer() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    void 
    start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    float 
    seconds() {
        std::chrono::duration<float> duration = std::chrono::high_resolution_clock::now() - start_;
        return duration.count();
    }
};

struct GPUTimer {
    cudaEvent_t start_, stop_;

    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void 
    start(cudaStream_t stream) {
        cudaEventRecord(start_, stream);
    }

    float 
    seconds(cudaStream_t stream) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }
};
