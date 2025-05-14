#include <kermac_cutensor.cuh>
#include <kermac.cuh>

int
main() {
    using namespace kermac;
    using T = f32;

    static const u64 BYTES = 5'000'000ull;

    TensorCoreMode tcm = TensorCoreMode::FP32;

    u64 M = 20;
    u64 N = 10;
    u64 K = 6;

    DeviceStackAllocator dsa(BYTES);
    HostStackAllocator hsa(BYTES);

    cudaStream_t primary_stream = 0;

    CUTensor cutensor;

    Philox philox(dsa, 123, primary_stream);
    
    DeviceMultiTensor<T> A(dsa, M, K); // M-major
    DeviceMultiTensor<T> B(dsa, N, K); // N-major
    DeviceMultiTensor<T> C(dsa, M, N); // M-major

    tensor_rng(philox, RNGType::UNIFORM, A.tensor, c_one<T>, c_zero<T>, primary_stream);
    tensor_rng(philox, RNGType::UNIFORM, B.tensor, c_one<T>, c_zero<T>, primary_stream);

    tensor_print(hsa, A.tensor, "A", primary_stream);
    tensor_print(hsa, B.tensor, "B", primary_stream);
    
    // Can do einsum contractions!
    cutensor_contraction(
        cutensor, 
        dsa, tcm, c_one<T>, 
        A, "mk",
        B, "nk",
        c_zero<T>,
        C, "mn",
        C, "mn",
        primary_stream
    );
   
    tensor_print(hsa, C.tensor, "C", primary_stream);
}