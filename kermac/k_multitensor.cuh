#pragma once

#pragma once

#include <kermac.cuh>

namespace kermac {

// Calculate the amount of modes based on which dimensions are 0
static
i32 
_calc_num_modes(
    u64 d0, 
    u64 d1, 
    u64 d2, 
    u64 d3
) {
    if (d3 != 0) return 4;
    if (d2 != 0) return 3;
    if (d1 != 0) return 2;
    return 1;
}

// Calculate the amount of columns the remaining extent
// translates to in the Tensor struct
static
u64
_calc_num_cols(
    u64 d0,
    u64 d1,
    u64 d2,
    u64 d3
) {
    d0 = d0 == 0ull ? 1 : d0;
    d1 = d1 == 0ull ? 1 : d1;
    d2 = d2 == 0ull ? 1 : d2;
    d3 = d3 == 0ull ? 1 : d3;
    return d1 * d2 * d3;
}

template <MemorySpace memory_space, class T>
struct MultiTensor {
//TODO: make it so singletons don't take up extra alignment memory
    Tensor<memory_space, T> tensor;
    i64 extent[4]; // i64 for cutensor
    i64 stride[4]; // i64 for cutensor
    i32 num_modes; // i32 for cutensor
    bool is_singular = false;

    MultiTensor(
        StackAllocator<memory_space> &sa,
        u64 d0, u64 d1, u64 d2, u64 d3
    ) : tensor(sa, d0, _calc_num_cols(d0, d1, d2, d3)),
        extent{
            static_cast<i64>(d0),
            static_cast<i64>(d1),
            static_cast<i64>(d2),
            static_cast<i64>(d3)
        },
        stride{
            1ll,
            d1 == 0ull ? 0ll : static_cast<i64>(tensor.ld_rows), 
            d2 == 0ull ? 0ll : static_cast<i64>(tensor.ld_rows * d1), 
            d3 == 0ull ? 0ll : static_cast<i64>(tensor.ld_rows * d1 * d2)
        },
        num_modes(_calc_num_modes(d0, d1, d2, d3))
    {}

    MultiTensor(
        StackAllocator<memory_space> &sa,
        u64 d0, u64 d1, u64 d2
    ) : MultiTensor(sa, d0, d1, d2, 0ull) {}

    MultiTensor(
        StackAllocator<memory_space> &sa,
        u64 d0, u64 d1
    ) : MultiTensor(sa, d0, d1, 0ull, 0ull) {}

    MultiTensor(
        StackAllocator<memory_space> &sa,
        u64 d0
    ) : MultiTensor(sa, d0, 0ull, 0ull, 0ull) {}

    MultiTensor(
        StackAllocator<memory_space> &sa
    ) : MultiTensor(sa, 1, 0ull, 0ull, 0ull)
    {
        this->is_singular = true;
    }

    // Reshape from Tensor to MultiTensor
    MultiTensor(
        Tensor<memory_space, T> &tensor_to_reshape,
        u64 d0, u64 d1, u64 d2, u64 d3
    ) : tensor(tensor_to_reshape, 0, d0, 0, _calc_num_cols(d0, d1, d2, d3)),
        extent{
            static_cast<i64>(d0),
            static_cast<i64>(d1),
            static_cast<i64>(d2),
            static_cast<i64>(d3)
        },
        stride{
            1ll,
            d1 == 0ull ? 0ll : static_cast<i64>(tensor.ld_rows), 
            d2 == 0ull ? 0ll : static_cast<i64>(tensor.ld_rows * d1), 
            d3 == 0ull ? 0ll : static_cast<i64>(tensor.ld_rows * d1 * d2)
        },
        num_modes(_calc_num_modes(d0, d1, d2, d3))
    {
        ASSERT( d0 == tensor_to_reshape.num_rows );
        ASSERT( tensor.num_cols == tensor_to_reshape.num_cols );
    }

    MultiTensor(
        Tensor<memory_space, T> &tensor_to_reshape,
        u64 d0, u64 d1, u64 d2
    ) : MultiTensor(tensor_to_reshape, d0, d1, d2, 0ull) {}

    MultiTensor(
        Tensor<memory_space, T> &tensor_to_reshape,
        u64 d0, u64 d1
    ) : MultiTensor(tensor_to_reshape, d0, d1, 0ull, 0ull) {}

    MultiTensor(
        Tensor<memory_space, T> &tensor_to_reshape,
        u64 d0
    ) : MultiTensor(tensor_to_reshape, d0, 0ull, 0ull, 0ull) {}

    // MultiTensor(
    //     MultiTensor<memory_space, T> &multi_tensor_to_reshape,
    //     u64 d0, u64 d1, u64 d2, u64 d3
    // ) : MultiTensor(multi_tensor_to_reshape.tensor, d0, d1, d2, d3)
    // {}

    // MultiTensor(
    //     MultiTensor<memory_space, T> &multi_tensor_to_reshape,
    //     u64 d0, u64 d1, u64 d2
    // ) : MultiTensor(multi_tensor_to_reshape.tensor, d0, d1, d2)
    // {}

    // MultiTensor(
    //     MultiTensor<memory_space, T> &multi_tensor_to_reshape,
    //     u64 d0, u64 d1
    // ) : MultiTensor(multi_tensor_to_reshape.tensor, d0, d1)
    // {}

    // MultiTensor(
    //     MultiTensor<memory_space, T> &multi_tensor_to_reshape,
    //     u64 d0
    // ) : MultiTensor(multi_tensor_to_reshape.tensor, d0)
    // {}

    ~MultiTensor() {}

    MultiTensor(const MultiTensor&) = delete;
    MultiTensor& operator=(const MultiTensor&) = delete;

    MultiTensor(MultiTensor&&) noexcept = delete;
    MultiTensor& operator=(MultiTensor&&) noexcept = delete;

    bool
    is_dry_run() {
        return tensor.is_dry_run();
    }

    T *ptr() {
        return tensor.ptr();
    }

    T *ptr(
        u64 batch
    ) {
        ASSERT( num_modes > 1 );
        
        u64 batch_stride = stride[num_modes - 1];
        return tensor.ptr() + batch * batch_stride;
    }

    void
    set_singleton(
        T v,
        cudaStream_t stream
    ) {
        ASSERT( is_singular );

        DRY_RETURN(tensor);

        cuCheck( cudaMemcpyAsync(ptr(), &v, sizeof(T), cudaMemcpyHostToDevice, stream) );
    }

    T 
    get_singleton(
        cudaStream_t stream
    ) {
        ASSERT( is_singular );

        DRY_COND(tensor) return 0;

        T v;
        cuCheck( cudaMemcpyAsync(&v, ptr(), sizeof(T), cudaMemcpyDeviceToHost, stream) );
        cuCheck( cudaStreamSynchronize(stream) );
        return v;
    }
};

template <class T>
using DeviceMultiTensor = MultiTensor<MemorySpace::Device, T>;

template <class T>
using HostMultiTensor = MultiTensor<MemorySpace::Host, T>;

template <MemorySpace memory_space, class T>
TensorSliceCoords
multi_tensor_batch_slice(
    MultiTensor<memory_space, T> &tensor_to_slice,
    u64 batch_num 
) {
    ASSERT( tensor_to_slice.num_modes == 3 || tensor_to_slice.num_modes == 2 );

    if (tensor_to_slice.num_modes == 3) {
        u64 N = tensor_to_slice.extent[0];
        u64 D = tensor_to_slice.extent[1];
        u64 B = tensor_to_slice.extent[2];

        ASSERT( batch_num < B );

        return TensorSliceCoords{
            .row_start = 0,
            .num_rows = N,
            .col_start = batch_num * D,
            .num_cols = D
        };
    } else if (tensor_to_slice.num_modes == 2) {
        u64 N = tensor_to_slice.extent[0];
        u64 B = tensor_to_slice.extent[1];

        ASSERT( batch_num < B );

        return TensorSliceCoords{
            .row_start = 0,
            .num_rows = N,
            .col_start = batch_num,
            .num_cols = 1
        };
    }

    ASSERT( false );
}

template <MemorySpace memory_space, class T>
void
multi_tensor_print_batches(
    HostStackAllocator &hsa,
    MultiTensor<memory_space, T> &multi_tensor,
    const char *multi_tensor_name,
    cudaStream_t stream,
    i32 edge_items = 6
) {
    // ASSERT( multi_tensor.num_modes == 3 || multi_tensor.num_modes == 2 );

    u64 B = multi_tensor.extent[multi_tensor.num_modes-1];
    for (u64 b = 0; b < B; b++) {
        DeviceTensor<T> multi_tensor_view(multi_tensor.tensor, multi_tensor_batch_slice(multi_tensor, b));
        WET_COND(multi_tensor) printf("(%" PRIu64 ") ", b);
        tensor_print(hsa, multi_tensor_view, multi_tensor_name, stream, edge_items);
    }
}

template <MemorySpace memory_space, class T>
void
multi_tensor_zip_print_batches(
    HostStackAllocator &hsa,
    MultiTensor<memory_space, T> &multi_tensor_a,
    const char *multi_tensor_a_name,
    MultiTensor<memory_space, T> &multi_tensor_b,
    const char *multi_tensor_b_name,
    cudaStream_t stream,
    i32 edge_items = 6
) {
    ASSERT( multi_tensor_a.num_modes == 3 || multi_tensor_a.num_modes == 2 );
    ASSERT( multi_tensor_b.num_modes == 3 || multi_tensor_b.num_modes == 2 );

    u64 B = multi_tensor_a.extent[multi_tensor_a.num_modes-1];

    ASSERT( B == multi_tensor_a.extent[multi_tensor_a.num_modes-1] );
    ASSERT( B == multi_tensor_b.extent[multi_tensor_b.num_modes-1] );

    for (u64 b = 0; b < B; b++) {
        DeviceTensor<T> multi_tensor_a_view(multi_tensor_a.tensor, multi_tensor_batch_slice(multi_tensor_a, b));
        printf("(%" PRIu64 ") ", b);
        tensor_print(hsa, multi_tensor_a_view, multi_tensor_a_name, stream, edge_items);

        DeviceTensor<T> multi_tensor_b_view(multi_tensor_b.tensor, multi_tensor_batch_slice(multi_tensor_b, b));
        printf("(%" PRIu64 ") ", b);
        tensor_print(hsa, multi_tensor_b_view, multi_tensor_b_name, stream, edge_items);
    }
}

// This is used for generating pointer offsets for gemm_batched.
// It takes a vector of indices from where to source from a.
// a is only used to obtain sizing information about the offsets.
// The offsets will contain the byte locations of the start of 
// each tensor the indices point to.
template <class T>
void
multi_tensor_indices_to_offset(
    HostStackAllocator &hsa,
    HostMultiTensor<u64> &indices, // B_run
    DeviceMultiTensor<T> &a, // .. ,B
    DeviceMultiTensor<T*> &offsets, // B_run
    cudaStream_t stream
) {
    ASSERT( indices.num_modes == 1 );
    ASSERT( offsets.num_modes == 1 );
    ASSERT( a.num_modes > 1 );

    u64 B = a.extent[a.num_modes-1];
    u64 B_run = indices.extent[0];

    ASSERT( B_run == indices.extent[0] );
    ASSERT( B_run == offsets.extent[0] );

    HostMultiTensor<T*> h_offsets(hsa, B_run);

    for (u64 b = 0; b < B_run; b++) {
        u64 batch_idx = indices.ptr()[b];
        ASSERT( batch_idx < B );
        // Get pointer at location in batch
        h_offsets.ptr()[b] = a.ptr(batch_idx);
    }

    set_matrix(h_offsets.tensor, offsets.tensor, stream);
}

}
