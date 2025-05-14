#include <stdio.h>

#include <kermac.cuh>
#include <kermac_cublas.cuh>

// Demonstration of kermac-cpp dealing with allocation
// for multiple concurrent streams

// This will be the same as 03-multi-stream but we'll allocate inside the loop.
// This creates more complexity because as each tensor leaves scope the memory is cleaned
// up. If the tensor is currently being worked on by a different stream, then the next
// stream will clobber this memory and reuse it.
// We need a method to create a stub allocator, use it in the current frame, record its largest
// offset and leave that memory alone until the streams synchronize again.

// Example is same as 03-multi-stream so comments are removed

int
main() {
    using namespace kermac;
    using T = f32;

    u64 M = 10;
    u64 N = 8;
    u64 K = 5;
    u64 num_batches = 3;
    u64 num_streams = num_batches;

    cudaStream_t primary_stream = 0;

    cudaStream_t streams[num_streams];
    for (i32 s = 0; s < num_streams; s++) {
        cuCheck( cudaStreamCreate(&streams[s]) );
    }

    cudaEvent_t primary_event;
    cuCheck( cudaEventCreate(&primary_event, cudaEventDisableTiming) );
    
    cudaEvent_t events[num_streams];
    for (i32 s = 0; s < num_streams; s++) {
        cuCheck( cudaEventCreate(&events[s], cudaEventDisableTiming) );
    }

    DeviceStackAllocator dsa;
    HostStackAllocator hsa;

    u64 max_device_memory_used_bytes = 0;
    u64 max_host_memory_used_bytes = 0;

    STACK_EXECUTE(dsa, hsa, {
        CUBlas cublas;
        Philox philox(dsa, 123, primary_stream);

        DeviceConstants<T> dcs(dsa, primary_stream);
        // This time only Cs is necessary as A and B for the respective 
        // batch is allocated inside the loop.
        DeviceMultiTensor<T> Cs(dsa, M, N, num_batches);

        // Still Record the primary_event in the primary_stream incase there was somehow work
        // issued before.
        cuCheck( cudaEventRecord(primary_event, primary_stream) );

        // This saves off the current offset in dsa 
        dsa.stream_snapshot();
        for (i32 b = 0; b < num_batches; b++) {
            // Starts recording offsets in this scope.
            // Each stream records its local largest_current_offset
            // So within each stream scope memory is reused locally.
            dsa.stream_alloc_start(); 
            {
                cudaStream_t this_stream = streams[b % num_streams];
                cuCheck( cudaStreamWaitEvent(this_stream, primary_event) );

                // Allocate new tensors
                DeviceTensor<T> A(dsa, M, K);
                DeviceTensor<T> B(dsa, N, K);
                // Slice this batch out of just Cs and make it a view.
                DeviceTensor<T> C_view(Cs.tensor, multi_tensor_batch_slice(Cs, b));

                // Set A and B directly in this stream
                tensor_rng(philox, RNGType::UNIFORM, A, c_one<T>, c_zero<T>, this_stream);
                tensor_rng(philox, RNGType::UNIFORM, B, c_one<T>, c_zero<T>, this_stream);

                cublas_gemm(
                    cublas,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    dcs.one,
                    A,
                    B,
                    dcs.zero,
                    C_view,
                    this_stream
                );
            }
            // Grabs the largest offset of the scope it just saw and keeps that
            // section active so the stream can work.
            dsa.stream_alloc_end();
        }
        // restores the state that was saved previously and returns all memory
        // allocated from the streams. It's this simple because all memory
        // is allocated in stack order so everything inside has to be dead.
        // Deletion of copy and move constructors means that a bunch of behaviors
        // that invalidate stack order are blocked such as assigning the tensor to 
        // a different name or returning the tensor from inside a function.
        dsa.stream_restore();

        for (i32 s = 0; s < num_streams; s++) {
            cuCheck( cudaEventRecord(events[s], streams[s]) );
            cuCheck( cudaStreamWaitEvent(primary_stream, events[s]) );
        }

        multi_tensor_print_batches(hsa, Cs, "Cs", primary_stream);

        // Store the memory stats to print later
        max_device_memory_used_bytes = dsa.largest_total_offset;
        max_host_memory_used_bytes = hsa.largest_total_offset;
    });

    for (i32 s = 0; s < num_streams; s++) {
        cuCheck( cudaEventDestroy(events[s]) );
    }

    cuCheck( cudaEventDestroy(primary_event) );

    for (i32 s = 0; s < num_streams; s++) {
        cuCheck( cudaStreamDestroy(streams[s]) );
    }

    // Note that this was all still executed DRY to record the amount of memory used
    // And then executed again with the following usage.
    // If the inside of the for loop used a routine that required workspace memory
    // such as cutensor or cusolver routines each stream would have its own memory
    // for this purpose protected from clobbering.
    printf("Device Memory: %" PRIu64 " bytes\n", max_device_memory_used_bytes);
    printf("Host Memory: %" PRIu64 " bytes\n", max_host_memory_used_bytes);
}