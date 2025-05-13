#include <kermac.cuh>
#include <kermac_cublas.cuh>

int
main() {
    using namespace kermac;
    using T = f32;

    static const u64 BYTES = 5'000'000ull;

    u64 M = 20;
    u64 N = 10;
    u64 K = 6;

    DeviceStackAllocator dsa(BYTES);
    HostStackAllocator hsa(BYTES);

    cudaStream_t primary_stream = 0;

    CUBlas cublas;

    Philox philox(dsa, 123, primary_stream);
    DeviceConstants<T> dcs(dsa, primary_stream);

    DeviceTensor<T> A(dsa, M, K); // M-major
    DeviceTensor<T> B(dsa, N, K); // N-major
    DeviceTensor<T> C(dsa, M, N); // M-major

    tensor_rng(philox, RNGType::UNIFORM, A, c_one<T>, c_zero<T>, primary_stream);
    tensor_rng(philox, RNGType::UNIFORM, B, c_one<T>, c_zero<T>, primary_stream);

    tensor_print(hsa, A, "A", primary_stream);
    tensor_print(hsa, B, "B", primary_stream);
    
    // Contracting on the slowest moving dimension, NT gemm
    cublas_gemm(
        cublas,
        CUBLAS_OP_N, 
        CUBLAS_OP_T,
        dcs.one,
        A,
        B,
        dcs.zero,
        C,
        primary_stream
    );

    tensor_print(hsa, C, "C", primary_stream);
}