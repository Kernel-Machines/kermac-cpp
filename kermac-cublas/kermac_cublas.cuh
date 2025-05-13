#pragma once

#include <cublas_v2.h>
#include <kermac.cuh>

namespace kermac {

struct CUBlas {
    cublasHandle_t handle;
    CUBlas() {
        cublasCheck( cublasCreate(&handle) );
        cublasCheck( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
        cublasCheck( cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) );
    }

    ~CUBlas() {
        cublasCheck( cublasDestroy(handle) );
    }

    CUBlas(const CUBlas&) = delete;
    CUBlas& operator=(const CUBlas&) = delete;

    CUBlas(CUBlas&&) = delete;
    CUBlas& operator=(CUBlas&&) = delete;

    void
    set_tf32() {
        cublasCheck( cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) );
    }

    void
    unset_tf32() {
        cublasCheck( cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) );
    }
};


template <class T>
void
cublas_extra_row_norm(
    CUBlas &cublas,
    cublasOperation_t trans_a, 
    cublasOperation_t trans_b,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceTensor<T> &b,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 N = MAX(c.num_rows, c.num_cols);
    u64 D = (trans_a == CUBLAS_OP_T) ? a.num_rows : a.num_cols;

    ASSERT( 1 == c.num_cols || 1 == c.num_rows);
    ASSERT( D == ((trans_b == CUBLAS_OP_T) ? b.num_cols : b.num_rows) );
    ASSERT( N == ((trans_a == CUBLAS_OP_T) ? a.num_cols : a.num_rows) );
    ASSERT( N == ((trans_b == CUBLAS_OP_T) ? b.num_rows : b.num_cols) );

    cublasOperation_t op_a = (trans_a == CUBLAS_OP_T) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = (trans_b == CUBLAS_OP_T) ? CUBLAS_OP_T : CUBLAS_OP_N;

    i64 stride_a = (trans_a == CUBLAS_OP_T) ? a.ld_rows : 1;
    i64 stride_b = (trans_b == CUBLAS_OP_T) ? 1 : b.ld_rows;
    i64 stride_c = (c.num_cols == 1) ? 1 : c.ld_rows;

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSgemmStridedBatched(
                cublas.handle,
                op_a,
                op_b,
                1, 1, D,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                stride_a,
                b.ptr(), b.ld_rows,
                stride_b,
                beta.ptr(),
                c.ptr(), c.ld_rows,
                stride_c,
                N
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgemmStridedBatched(
                cublas.handle,
                op_a,
                op_b,
                1, 1, D,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                stride_a,
                b.ptr(), b.ld_rows,
                stride_b,
                beta.ptr(),
                c.ptr(), c.ld_rows,
                stride_c,
                N
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void
cublas_extra_outer_product(
    CUBlas &cublas,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceTensor<T> &b,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 D = a.num_rows;
    u64 N = a.num_cols;
    u64 C = c.num_cols;

    ASSERT( N == b.num_rows );
    ASSERT( C == b.num_cols );
    ASSERT( N*D == c.num_rows );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSgemmStridedBatched(
                cublas.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                D, C, 1,
                alpha.ptr(),
                a.ptr(), a.ld_rows, a.ld_rows,
                b.ptr(), b.ld_rows, 1,
                beta.ptr(),
                c.ptr(), c.ld_rows, D,
                N
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgemmStridedBatched(
                cublas.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                D, C, 1,
                alpha.ptr(),
                a.ptr(), a.ld_rows, a.ld_rows,
                b.ptr(), b.ld_rows, 1,
                beta.ptr(),
                c.ptr(), c.ld_rows, D,
                N
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void
cublas_extra_broadcast_batched_gemm(
    CUBlas &cublas,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceTensor<T> &b,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 V_N = b.num_rows;
    u64 H_N = b.num_cols;
    u64 C = a.num_cols;
    u64 D = a.num_rows / V_N;

    ASSERT( a.num_rows == V_N * D );
    ASSERT( c.num_cols == C );
    ASSERT( c.num_rows == H_N * D );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSgemmStridedBatched(
                cublas.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                D, H_N, V_N,
                alpha.ptr(),
                a.ptr(), D, a.ld_rows,
                b.ptr(), b.ld_rows, 0,
                beta.ptr(),
                c.ptr(), D, c.ld_rows,
                C
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgemmStridedBatched(
                cublas.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                D, H_N, V_N,
                alpha.ptr(),
                a.ptr(), D, a.ld_rows,
                b.ptr(), b.ld_rows, 0,
                beta.ptr(),
                c.ptr(), D, c.ld_rows,
                C
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void
cublas_extra_batched_syrk_full(
    CUBlas &cublas,
    u64 D,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 H_N = a.num_rows / D;
    u64 C = c.num_cols;

    ASSERT( a.num_rows == H_N * D );
    ASSERT( a.num_cols == C );
    ASSERT( c.num_rows == D * D );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSgemmStridedBatched(
                cublas.handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                D, D, H_N,
                alpha.ptr(),
                a.ptr(), D, a.ld_rows,
                a.ptr(), D, a.ld_rows,
                beta.ptr(),
                c.ptr(), D, c.ld_rows,
                C
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgemmStridedBatched(
                cublas.handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                D, D, H_N,
                alpha.ptr(),
                a.ptr(), D, a.ld_rows,
                a.ptr(), D, a.ld_rows,
                beta.ptr(),
                c.ptr(), D, c.ld_rows,
                C
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void 
cublas_nrm2(
    CUBlas &cublas,
    DeviceTensor<T> &a,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
    u64 M = a.num_rows;

    ASSERT( 1 == a.num_cols );
    ASSERT( 1 == c.num_rows );
    ASSERT( 1 == c.num_cols );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSnrm2(
                cublas.handle,
                M,
                a.ptr(), 1,
                c.ptr()
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgemv(
                cublas.handle,
                M,
                a.ptr(), 1,
                c.ptr()
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void 
cublas_gemv(
    CUBlas &cublas,
    cublasOperation_t trans, 
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceTensor<T> &b,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 M = c.num_rows;
    u64 N = c.num_cols;
    u64 K = (trans == CUBLAS_OP_T) ? a.num_rows : a.num_cols;

    ASSERT( M == ((trans == CUBLAS_OP_T) ? a.num_cols : a.num_rows) );
    ASSERT( K == ((trans == CUBLAS_OP_T) ? a.num_rows : a.num_cols) );

    ASSERT( K == b.num_rows );
    ASSERT( N == 1 );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSgemv(
                cublas.handle, trans,
                M, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                b.ptr(), 1,
                beta.ptr(),
                c.ptr(), 1
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgemv(
                cublas.handle, trans,
                M, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                b.ptr(), 1,
                beta.ptr(),
                c.ptr(), 1
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void 
cublas_gemm(
    CUBlas &cublas,
    cublasOperation_t trans_a, 
    cublasOperation_t trans_b,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceTensor<T> &b,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 M = c.num_rows;
    u64 N = c.num_cols;
    u64 K = (trans_a == CUBLAS_OP_T) ? a.num_rows : a.num_cols;

    ASSERT( M == ((trans_a == CUBLAS_OP_T) ? a.num_cols : a.num_rows) );
    ASSERT( K == ((trans_a == CUBLAS_OP_T) ? a.num_rows : a.num_cols) );

    ASSERT( N == ((trans_b == CUBLAS_OP_T) ? b.num_rows : b.num_cols) );
    ASSERT( K == ((trans_b == CUBLAS_OP_T) ? b.num_cols : b.num_rows) );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSgemm(
                cublas.handle, trans_a, trans_b,
                M, N, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                b.ptr(), b.ld_rows,
                beta.ptr(),
                c.ptr(), c.ld_rows
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgemm(
                cublas.handle, trans_a, trans_b,
                M, N, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                b.ptr(), b.ld_rows,
                beta.ptr(),
                c.ptr(), c.ld_rows
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void
cublas_syrk(
    CUBlas &cublas,
    cublasOperation_t trans,
    cublasFillMode_t uplo,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 N = c.num_rows;
    u64 K = (trans == CUBLAS_OP_T) ? a.num_rows : a.num_cols;

    ASSERT( N == c.num_rows );
    ASSERT( N == c.num_cols );
    ASSERT( N == ((trans == CUBLAS_OP_T) ? a.num_cols : a.num_rows) );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSsyrk(
                cublas.handle, uplo, trans,
                N, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                beta.ptr(),
                c.ptr(), c.ld_rows
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDsyrk(
                cublas.handle, uplo, trans,
                N, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                beta.ptr(),
                c.ptr(), c.ld_rows
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void
cublas_syrkx(
    CUBlas &cublas,
    cublasOperation_t trans,
    cublasFillMode_t uplo,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceTensor<T> &b,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 N = c.num_rows;
    u64 K = (trans == CUBLAS_OP_T) ? a.num_rows : a.num_cols;
    
    ASSERT( K == ((trans == CUBLAS_OP_T) ? a.num_rows : a.num_cols) );
    ASSERT( K == ((trans == CUBLAS_OP_T) ? b.num_rows : b.num_cols) );

    ASSERT( N == c.num_rows );
    ASSERT( N == c.num_cols );
    ASSERT( N == ((trans == CUBLAS_OP_T) ? a.num_cols : a.num_rows) );
    ASSERT( N == ((trans == CUBLAS_OP_T) ? b.num_cols : b.num_rows) );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSsyrkx(
                cublas.handle, uplo, trans,
                N, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                b.ptr(), b.ld_rows,
                beta.ptr(),
                c.ptr(), c.ld_rows
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDsyrkx(
                cublas.handle, uplo, trans,
                N, K,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                b.ptr(), b.ld_rows,
                beta.ptr(),
                c.ptr(), c.ld_rows
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void 
cublas_geam(
    CUBlas &cublas,
    cublasOperation_t trans_a, 
    cublasOperation_t trans_b,
    DeviceConstant<T> &alpha,
    DeviceTensor<T> &a,
    DeviceConstant<T> &beta,
    DeviceTensor<T> &b,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 M = c.num_rows;
    u64 N = c.num_cols;

    ASSERT( M == ((trans_a == CUBLAS_OP_T) ? a.num_cols : a.num_rows) );
    ASSERT( N == ((trans_a == CUBLAS_OP_T) ? a.num_rows : a.num_cols) );

    ASSERT( M == ((trans_b == CUBLAS_OP_T) ? b.num_cols : b.num_rows) );
    ASSERT( N == ((trans_b == CUBLAS_OP_T) ? b.num_rows : b.num_cols) );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );

    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSgeam(
                cublas.handle, trans_a, trans_b,
                M, N,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                beta.ptr(),
                b.ptr(), b.ld_rows,
                c.ptr(), c.ld_rows
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDgeam(
                cublas.handle, trans_a, trans_b,
                M, N,
                alpha.ptr(),
                a.ptr(), a.ld_rows,
                beta.ptr(),
                b.ptr(), b.ld_rows,
                c.ptr(), c.ld_rows
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void 
cublas_dgmm(
    CUBlas &cublas,
    cublasSideMode_t mode,
    DeviceTensor<T> &a,
    DeviceTensor<T> &diag,
    DeviceTensor<T> &c,
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    u64 M = c.num_rows;
    u64 N = c.num_cols;

    u64 diag_dim = MAX(diag.num_rows, diag.num_cols);

    ASSERT( 1 == diag.num_rows || 1 == diag.num_cols );
    ASSERT( diag_dim == ((mode == CUBLAS_SIDE_LEFT) ? M : N) );

    ASSERT( M == a.num_rows );
    ASSERT( N == a.num_cols );

    i64 stride_diag = (diag.num_cols == 1) ? 1 : diag.ld_rows;

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );
    
    if constexpr (std::is_same_v<T, f32>) {
        cublasCheck(
            cublasSdgmm(cublas.handle, mode,
                M, N,
                a.ptr(), a.ld_rows,
                diag.ptr(), stride_diag,
                c.ptr(), c.ld_rows
            )
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasCheck(
            cublasDdgmm(cublas.handle, mode,
                M, N,
                a.ptr(), a.ld_rows,
                diag.ptr(), stride_diag,
                c.ptr(), c.ld_rows
            )
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

template <class T>
void
cublas_gemm_batched(
    CUBlas &cublas,
    cublasOperation_t trans_a,
    cublasOperation_t trans_b,
    DeviceConstant<T> &alpha,
    DeviceMultiTensor<T> &a,            // M,K,B
    DeviceMultiTensor<T*> &offsets_a,  // B_run
    DeviceMultiTensor<T> &b,            // K,N,B
    DeviceMultiTensor<T*> &offsets_b,  // B_run
    DeviceConstant<T> &beta,
    DeviceMultiTensor<T> &c,            // M,N,B
    DeviceMultiTensor<T*> &offsets_c,  // M,N,B_run
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    ASSERT( a.num_modes == 3 );
    ASSERT( offsets_a.num_modes == 1 );

    ASSERT( b.num_modes == 3 );
    ASSERT( offsets_b.num_modes == 1 );

    ASSERT( c.num_modes == 3 );
    ASSERT( offsets_c.num_modes == 1 );

    u64 M = c.extent[0];
    u64 N = c.extent[1];
    u64 K = (trans_a == CUBLAS_OP_T) ? a.extent[0] : a.extent[1];
    u64 B = a.extent[2];
    u64 B_run = offsets_a.extent[0];

    ASSERT( M == ((trans_a == CUBLAS_OP_T) ? a.extent[1] : a.extent[0]) );
    ASSERT( K == ((trans_a == CUBLAS_OP_T) ? a.extent[0] : a.extent[1]) );

    ASSERT( N == ((trans_b == CUBLAS_OP_T) ? b.extent[0] : b.extent[1]) );
    ASSERT( K == ((trans_b == CUBLAS_OP_T) ? b.extent[1] : b.extent[0]) );

    ASSERT( B == a.extent[2] );
    ASSERT( B == b.extent[2] );
    ASSERT( B_run == c.extent[2] );

    ASSERT( B_run == offsets_a.extent[0] );
    ASSERT( B_run == offsets_b.extent[0] );
    ASSERT( B_run == offsets_c.extent[0] );

    DRY_RETURN(a);

    cudaStream_t prev_stream;
    cublasCheck( cublasGetStream(cublas.handle, &prev_stream) );
    cublasCheck( cublasSetStream(cublas.handle, stream) );
    
    if constexpr (std::is_same_v<T,f32>) {
        cublasSgemmBatched(
            cublas.handle,
            trans_a, trans_b,
            M, N, K,
            alpha.ptr(),
            reinterpret_cast<f32**>(offsets_a.ptr()), a.stride[1],
            reinterpret_cast<f32**>(offsets_b.ptr()), b.stride[1],
            beta.ptr(),
            reinterpret_cast<f32**>(offsets_c.ptr()), c.stride[1],
            B_run
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        cublasDgemmBatched(
            cublas.handle,
            trans_a, trans_b,
            M, N, K,
            alpha.ptr(),
            reinterpret_cast<f64**>(offsets_a.ptr()), a.stride[1],
            reinterpret_cast<f64**>(offsets_b.ptr()), b.stride[1],
            beta.ptr(),
            reinterpret_cast<f64**>(offsets_c.ptr()), c.stride[1],
            B_run
        );
    }

    cublasCheck( cublasSetStream(cublas.handle, prev_stream) );
}

}
