#include <stdio.h>

#include <kermac.cuh>
#include <kermac_cublas.cuh>

// Demonstration of kermac-cpp library dynamic memory

// We can create a series of operations with an unallocated stack alloctor.
// This allocator will track all the allocations and record its largest memory offset
// during the entire run. Akin to audio plugins we call this DRY/WET. The macro
// DRY_RETURN(a) does an if check for a.is_dry_run() which every struct which uses
// the allocator has a definition for. You can use this and WET_COND(a) to block
// the usage of data while the allocator is in a DRY state.

// Declare a function that we'll run twice. 
// Once to capture the memory usage.
// Once again to run after we've allocated.
template <class T>
void
func(
    u64 M, u64 N, u64 K,
    kermac::HostStackAllocator &hsa,    // kermac Objects need to be passed by reference
    kermac::DeviceStackAllocator &dsa   // because Copy, Move constructors are all deleted.
) {
    using namespace kermac;

    cudaStream_t primary_stream = 0; 

    CUBlas cublas;
    Philox philox(dsa, 123, primary_stream);

    // Make device constants for some frequently used constants.
    // Cublas has device pointer mode set so it requires all constants to be allocated in
    // GPU memory. The copy happens on the stream thats passed in.
    DeviceConstants<T> dcs(dsa, primary_stream);

    // Can make a different constant by doing
    // It needs to be on a stream to synchronize the copy.
    DeviceConstant<T> negative_two(dsa, -2.0, primary_stream);

    DeviceMultiTensor<T> A(dsa, M, K);
    DeviceMultiTensor<T> B(dsa, N, K);
    DeviceMultiTensor<T> C(dsa, M, N);

    tensor_rng(philox, RNGType::UNIFORM, A.tensor, c_one<T>, c_zero<T>, primary_stream);

    T B_scale = 1.1;
    tensor_rng(philox, RNGType::UNIFORM, B.tensor, B_scale, c_zero<T>, primary_stream);

    // cublas_gemm and all other routines will only exectute their sanity checks
    // and allocate the necessary memory when the allocator is DRY.
    // When the allocator is WET the actual routine will run.
    // If a routine does allocate when allocator is in DRY state, it 
    // will only record the memory offset.
    cublas_gemm(
        cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dcs.one,
        A.tensor,
        B.tensor,
        dcs.zero,
        C.tensor, 
        primary_stream
    );

    // Printing has a check for dry run so it doesn't print if the allocator is
    // doing its dry run.
    tensor_print(hsa, C.tensor, "C", primary_stream);
}


int
main() {
    using namespace kermac;
    using T = f32;

    {
        // Declare allocators without any bytes in this scope only.
        HostStackAllocator hsa;
        DeviceStackAllocator dsa;

        u64 M = 10;
        u64 N = 5;
        u64 K = 3;

        // We run func once with the allocators DRY
        func<T>(M, N, K, hsa, dsa);

        // Allocators now have a `largest_total_offset` so now we
        // know how much memory is required to execute func<T>
        printf("Host Memory Needed: %" PRIu64 " bytes\n", hsa.largest_total_offset);
        printf("Device Memory Needed: %" PRIu64 " bytes\n", dsa.largest_total_offset);

        // alloc() will allocate the amount of memory they saw during their 
        // journey through func<T> world.
        // These bytes reflect all of the necessary alignment of the tensors and 
        // the padding on the leading dimensions.
        hsa.alloc();
        dsa.alloc();

        // Now we can run func<T> again. With the allocators full.
        func<T>(M, N, K, hsa, dsa);
    }

    // Now hsa and dsa fall out of scope and their memory is cleaned up.

    // We can try this again with the STACK_EXECUTE macro
    {
        DeviceStackAllocator dsa;
        HostStackAllocator hsa;

        u64 M = 10;
        u64 N = 5;
        u64 K = 3;

        // We can run this to run the func<T> function twice.
        // It will run the allocators DRY and then WET.
        // At the end the allocators are cleared (freed)
        STACK_EXECUTE(dsa, hsa, {
            func<T>(M, N, K, hsa, dsa);
        });

        // We can also use STACK_EXECUTE to declare 
        // routines inline. This is func<T> pasted inside
        // STACK_EXECUTE. Again this will run the
        // allocators DRY and then WET.
        STACK_EXECUTE(dsa, hsa, {
                using namespace kermac;

                cudaStream_t primary_stream = 0; 

                CUBlas cublas;
                Philox philox(dsa, 123, primary_stream);

                // Make device constants for some frequently used constants.
                // Cublas has device pointer mode set so it requires all constants to be allocated in
                // GPU memory. The copy happens on the stream thats passed in.
                DeviceConstants<T> dcs(dsa, primary_stream);

                // Can make a different constant by doing
                // It needs to be on a stream to synchronize the copy.
                DeviceConstant<T> negative_two(dsa, -2.0, primary_stream);

                DeviceMultiTensor<T> A(dsa, M, K);
                DeviceMultiTensor<T> B(dsa, N, K);
                DeviceMultiTensor<T> C(dsa, M, N);

                tensor_rng(philox, RNGType::UNIFORM, A.tensor, c_one<T>, c_zero<T>, primary_stream);

                T B_scale = 1.1;
                tensor_rng(philox, RNGType::UNIFORM, B.tensor, B_scale, c_zero<T>, primary_stream);

                // cublas_gemm and all other routines will only exectute their sanity checks
                // and allocate the necessary memory when the allocator is DRY.
                // When the allocator is WET the actual routine will run.
                // If a routine does allocate when allocator is in DRY state, it 
                // will only record the memory offset.
                cublas_gemm(
                    cublas,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    dcs.one,
                    A.tensor,
                    B.tensor,
                    dcs.zero,
                    C.tensor, 
                    primary_stream
                );

                // Printing has a check for dry run so it doesn't print if the allocator is
                // doing its dry run.
                tensor_print(hsa, C.tensor, "C", primary_stream);
        });
    }
}