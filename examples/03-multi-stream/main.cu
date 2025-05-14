#include <stdio.h>

#include <kermac.cuh>
#include <kermac_cublas.cuh>

// Demonstration of kermac-cpp dealing with  multiple concurrent streams
int
main() {
    using namespace kermac;
    using T = f32;

    u64 M = 10;
    u64 N = 8;
    u64 K = 5;
    u64 num_batches = 3;  // num batches
    u64 num_streams = num_batches; // one stream per batch

    cudaStream_t primary_stream = 0;

    // one stream per batch, create the streams
    cudaStream_t streams[num_streams];
    for (i32 s = 0; s < num_streams; s++) {
        cuCheck( cudaStreamCreate(&streams[s]) );
    }

    // We need one primary_event to signal to the other 
    // streams that the primary event is done and the 
    // other streams can start their work.
    cudaEvent_t primary_event;
    cuCheck( cudaEventCreate(&primary_event, cudaEventDisableTiming) );
    
    // Need one event per stream to signal to the
    // primary stream that they're done
    cudaEvent_t events[num_streams];
    for (i32 s = 0; s < num_streams; s++) {
        // We don't want timing, its slower.
        cuCheck( cudaEventCreate(&events[s], cudaEventDisableTiming) );
    }

    DeviceStackAllocator dsa;
    HostStackAllocator hsa;

    // Create a dynamic allocator frame
    STACK_EXECUTE(dsa, hsa, {
        CUBlas cublas;
        Philox philox(dsa, 123, primary_stream);

        DeviceConstants<T> dcs(dsa, primary_stream);
        DeviceMultiTensor<T> As(dsa, M, K, num_batches);
        DeviceMultiTensor<T> Bs(dsa, N, K, num_batches);
        DeviceMultiTensor<T> Cs(dsa, M, N, num_batches);

        tensor_rng(philox, RNGType::UNIFORM, As.tensor, c_one<T>, c_zero<T>, primary_stream);
        tensor_rng(philox, RNGType::UNIFORM, Bs.tensor, c_one<T>, c_zero<T>, primary_stream);

        // Record the primary_event in the primary_stream to mark the end
        // of setting the rngs values on As and Bs
        cuCheck( cudaEventRecord(primary_event, primary_stream) );

        for (i32 b = 0; b < num_batches; b++) {
            // b % num_streams lets us cycle through batches regardless of how many streams
            cudaStream_t this_stream = streams[b % num_streams];
            // Make this stream wait for the primary_event to finish
            // to known that the rng values are set.
            cuCheck( cudaStreamWaitEvent(this_stream, primary_event) );

            // Slice this batch out of each of the tensors and make it a view.
            DeviceTensor<T> A_view(As.tensor, multi_tensor_batch_slice(As, b));
            DeviceTensor<T> B_view(Bs.tensor, multi_tensor_batch_slice(Bs, b));
            DeviceTensor<T> C_view(Cs.tensor, multi_tensor_batch_slice(Cs, b));

            // Now do a cublas_gemm call for each of the sliced out batches on the given stream.
            cublas_gemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                dcs.one,
                A_view,
                B_view,
                dcs.zero,
                C_view,
                this_stream
            );
        }

        // Now issue the events for each stream that marks the completion of this work.
        // Each stream can issue for multiple batches, which is why the events must be issued
        // at the end of the loop.
        // The primary_stream also schedules waits for the respective event from each stream
        for (i32 s = 0; s < num_streams; s++) {
            cuCheck( cudaEventRecord(events[s], streams[s]) );
            cuCheck( cudaStreamWaitEvent(primary_stream, events[s]) );
        }

        // The primary_stream has waited for each stream to complete and it 
        // is now safe to print the result.
        multi_tensor_print_batches(hsa, Cs, "Cs", primary_stream);

    });

    for (i32 s = 0; s < num_streams; s++) {
        cuCheck( cudaEventDestroy(events[s]) );
    }

    cuCheck( cudaEventDestroy(primary_event) );

    // Delete the created streams
    for (i32 s = 0; s < num_streams; s++) {
        cuCheck( cudaStreamDestroy(streams[s]) );
    }
}