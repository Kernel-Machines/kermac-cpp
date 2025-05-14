#include <kermac.cuh>
#include <kermac_cublas.cuh>
#include <kermac_cusolver.cuh>

int
main() {
    using namespace kermac;
    using T = f32;

    static const u64 BYTES = 5'000'000ull;

    u64 N = 20;

    DeviceStackAllocator dsa(BYTES);
    HostStackAllocator hsa(BYTES);

    cudaStream_t primary_stream = 0;

    CUBlas cublas;
    CUSolver cusolver;

    Philox philox(dsa, 123, primary_stream);
    DeviceConstants<T> dcs(dsa, primary_stream);

    DeviceTensor<T> A(dsa, N, N); // M-major
    DeviceTensor<T> W(dsa, N, 1); // M-major

    tensor_rng(philox, RNGType::UNIFORM, A, c_one<T>, c_zero<T>, primary_stream);

    DeviceConstant<i32> info(dsa, c_negative_one<T>, primary_stream);
    cusolver_syevd(
        cusolver,
        hsa,
        dsa,
        CUBLAS_FILL_MODE_UPPER,
        info,
        A,
        W,
        primary_stream
    );

    printf("Cusolver info %d (Should be 0 if success)\n", info.get(primary_stream));

    printf("Eigenvectors:\n");
    tensor_print(hsa, A, "A", primary_stream);
    printf("\n");
    
    printf("Eigenvalues:\n");
    tensor_print(hsa, W, "W", primary_stream);
    printf("\n");
}