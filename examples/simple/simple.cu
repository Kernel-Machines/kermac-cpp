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
    u64 num_batches = 2;  // num batches

    // Create Stack Allocators in this scope with `BYTES` allocated for each.
    // Memory is aligned to `HOST_ALLOC_ALIGNMENT` and `DEVICE_ALLOC_ALIGNMENT` respectively.
    // Each allocation from these just moves a pointer to the `NEAREST_LARGER_MULTIPLE` of the alignment.
    // Allocation and Deallocation is checked for sanity by a counter.
    HostStackAllocator hsa(BYTES);
    DeviceStackAllocator dsa(BYTES);

    // Declare a multiple dimension tensor that will only survive in this scope
    // M is stride 1, B is slowest moving dimension.
    // Leftmost dimension is padded to `TENSOR_LD_ALIGNMENT_BYTES`. 
    // stride[0] is 1, 
    // stride[1] is ldM (padded to `TENSOR_LD_ALIGNMENT_BYTES`)
    // stride[2] is N * ldM (NOT N * M)
    DeviceMultiTensor<T> A(dsa, M, N, num_batches);

    // Get default cuda stream
    cudaStream_t primary_stream = 0;

    // Initialize the philox state swap buffer, syncs on `primary_stream`. Uses memory from `dsa`
    Philox philox(dsa, 123, primary_stream);

    // use `A` as a 2D tensor of M rows by N * B columns. Uniform rng, scale by 1.0 (casted to T) and shifted by 0.0
    // launch the cuda rng kernel on stream `primary_stream`
    tensor_rng(philox, RNGType::UNIFORM, A.tensor, c_one<T>, c_zero<T>, primary_stream);

    // Print the tensor as a 2D tensor of M rows by N * B columns. 
    // `hsa` is passed in to allocate on the host and copy the device memory to host synchronizing on the `primary_stream`
    printf("Full A as a 2D tensor with rng values:\n");
    tensor_print(hsa, A.tensor, "A", primary_stream);
    printf("\n");

    // Do this all again to demonstrate that the philox generator state updated.
    tensor_rng(philox, RNGType::UNIFORM, A.tensor, c_one<T>, c_zero<T>, primary_stream);
    printf("Full A as a 2D tensor with different rng values:\n");
    tensor_print(hsa, A.tensor, "A", primary_stream);
    printf("\n");

    // Get coordinates that slice out the first batch of the multi tensor
    TensorSliceCoords tsc = multi_tensor_batch_slice(A, 0);

    // A_view is now a view of A
    // This view has the same alloc and dealloc counters as a full memory allocation for sanity
    // However it doesn't get freed from the stack allocator when out of scope.
    DeviceTensor<T> A_view(A.tensor, tsc);

    // Print just the view
    printf("View of A:\n");
    tensor_print(hsa, A_view, "A", primary_stream);
    printf("\n");

    // Set just the view to zeros
    tensor_set_zero(A_view, primary_stream);

    // Print the full tensor showing that the first batch are now all zeros
    printf("Full A as a 2D tensor:\n");
    tensor_print(hsa, A.tensor, "A", primary_stream);
    printf("\n");

    printf("Batches of A multitensor:\n");
    multi_tensor_print_batches(hsa, A, "A", primary_stream);
    printf("\n");

    {
        // Allocating a tensor inside of a scope makes the memory available only while in scope.
        // After leaving scope the stack allocator moves its allocation pointer back and checks 
        // to make sure the counters it expects are sane.
        // If cuda kernels are launched on different streams, then when this scope is left and 
        // a new tensor is allocated it might clobber this memory.
        // If a cuda event is issued on the work done on tensor B then the alternate stream can sync
        // on that event before doing the work and everything should work out fine.
        DeviceMultiTensor<T> B(dsa, M, N, num_batches);
        // Set B multitensor to rng values, scaled by two and shifted by negative_one
        tensor_rng(philox, RNGType::UNIFORM, B.tensor, c_two<T>, c_negative_one<T>, primary_stream);

        // Convenience function to compare two tensors by eye for each batch one-to-one
        printf("Print the batches of A and B together:\n");
        multi_tensor_zip_print_batches(hsa, A, "A", B, "B", primary_stream);
        printf("\n");
    }

    // B is now gone.
    // C will clobber B's memory. It is important to zero out the memory if it's required to be zero.
    u64 M_sub = 2;
    u64 N_sub = 3;
    DeviceMultiTensor<T> C(dsa, M_sub, N_sub, num_batches);

    printf("C now contains some of B:\n");
    tensor_print(hsa, C.tensor, "C", primary_stream);
    printf("\n");

    // Set C to zeros, issue on primary stream so it doesn't
    // clobber B while B's work might still be in progress
    tensor_set_zero(C.tensor, primary_stream);

    printf("C now contains zeros:\n");
    tensor_print(hsa, C.tensor, "C", primary_stream);
    printf("\n");

    // After this scope is left the stack allocators are cleaned up.
}