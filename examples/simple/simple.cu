#include <stdio.h>

#include <kermac.cuh>

// Simple demonstration of kermac-cpp library usage

int
main() {
    using namespace kermac;
    using T = f32;

    static const u64 BYTES = 5'000'000ull;

    u64 M = 3; // num rows
    u64 N = 5; // num cols
    u64 B = 2;  // num batches

    // Create Stack Allocators in this scope with `BYTES` allocated for each
    // Memory is aligned to `HOST_ALLOC_ALIGNMENT` and `DEVICE_ALLOC_ALIGNMENT` respectively
    // each allocation from these just moves a pointer to the `NEAREST_LARGER_MULTIPLE` of the alignment
    // allocation and deallocation is checked for sanity by a counter.
    HostStackAllocator hsa(BYTES);
    DeviceStackAllocator dsa(BYTES);

    // Declare a multiple dimension tensor that will only survive in this scope
    // M is stride 1, B is slowest moving dimension.
    // Leftmost dimension is padded to `TENSOR_LD_ALIGNMENT_BYTES`. 
    // stride[0] is 1, 
    // stride[1] is ldM (padded to `TENSOR_LD_ALIGNMENT_BYTES`)
    // stride[2] is N * ldM (NOT N * M)
    DeviceMultiTensor<T> A(dsa, M, N, B);

    // Get default cuda stream
    cudaStream_t primary_stream = 0;

    // Initialize the philox state swap buffer, syncs on `primary_stream`. Uses memory from `dsa`
    Philox philox(dsa, 123, primary_stream);

    // use `A` as a 2D tensor of M rows by N * B columns. Uniform rng, scale by 1.0 (casted to T) and shifted by 0.0
    // launch the cuda rng kernel on stream `primary_stream`
    tensor_rng(philox, RNGType::UNIFORM, A.tensor, c_one<T>, c_zero<T>, primary_stream);

    // Print the tensor as a 2D tensor of M rows by N * B columns. 
    // `hsa` is passed in to allocate on the host and copy the device memory to host synchronizing on the `primary_stream`
    tensor_print(hsa, A.tensor, primary_stream);

    printf("\n");

    // Do this all again to demonstrate that the philox generator state updated.
    tensor_rng(philox, RNGType::UNIFORM, A.tensor, c_one<T>, c_zero<T>, primary_stream);
    tensor_print(hsa, A.tensor, primary_stream);
}