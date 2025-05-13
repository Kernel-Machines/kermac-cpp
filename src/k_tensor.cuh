#pragma once

#include <kermac.cuh>

namespace kermac {

static constexpr u64 TENSOR_ALIGNMENT_BYTES = DEVICE_ALLOC_ALIGNMENT;
static constexpr u64 TENSOR_LD_ALIGNMENT_BYTES = 128ull;

struct TensorSliceCoords {
    u64 row_start;
    u64 num_rows;
    u64 col_start;
    u64 num_cols;
};

template <class T>
static
u64 
_tensor_calc_ld_rows(
    u64 num_rows
) {
    u64 data_type_alignment = TENSOR_LD_ALIGNMENT_BYTES / sizeof(T);
    return data_type_alignment * NEAREST_LARGER_MULTIPLE(num_rows, data_type_alignment);
}

template <MemorySpace memory_space, class T>
struct Tensor {
    const u64 num_rows;
    const u64 num_cols;
    const u64 ld_rows;
    Memory<memory_space> memory;

    Tensor(
        StackAllocator<memory_space> &sa,
        u64 num_rows, 
        u64 num_cols
    ) : num_rows(num_rows),
        num_cols(num_cols),
        ld_rows(_tensor_calc_ld_rows<T>(num_rows)),
        memory(
            Memory<memory_space>(
                sa,
                num_cols * ld_rows * sizeof(T), 
                TENSOR_ALIGNMENT_BYTES
            )
        )
    {}

    Tensor(
        Tensor<memory_space, T> &tensor_to_view,
        u64 row_start, u64 num_rows,
        u64 col_start, u64 num_cols
    ) : num_rows(num_rows),
        num_cols(num_cols),
        ld_rows(tensor_to_view.ld_rows),
        memory(
            tensor_to_view.memory, (col_start * ld_rows + row_start) * sizeof(T)
        )
    {
        ASSERT( row_start < tensor_to_view.num_rows );
        ASSERT( col_start < tensor_to_view.num_cols );

        // Maybe let number of rows run past the limit?
        ASSERT( row_start + num_rows < tensor_to_view.num_rows + 1 );
        ASSERT( col_start + num_cols < tensor_to_view.num_cols + 1 );
    }

    Tensor(
        Tensor<memory_space, T> &tensor_to_view,
        TensorSliceCoords tensor_slice_coords
    ) : Tensor(
            tensor_to_view,
            tensor_slice_coords.row_start,
            tensor_slice_coords.num_rows,
            tensor_slice_coords.col_start,
            tensor_slice_coords.num_cols
        )
    {}

    ~Tensor() {}

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept = delete;
    Tensor& operator=(Tensor&&) noexcept = delete;

    bool
    is_dry_run() {
        return memory.is_dry_run();
    }

    T *ptr() {
        return reinterpret_cast<T*>(memory.ptr());
    }
};

template <class T>
using DeviceTensor = Tensor<MemorySpace::Device, T>;

template <class T>
using HostTensor = Tensor<MemorySpace::Host, T>;

template <class T>
void 
get_matrix(
    DeviceTensor<T> &src,
    HostTensor<T> &dst,
    cudaStream_t stream
) {
    ASSERT( src.num_rows == dst.num_rows );
    ASSERT( src.num_cols == dst.num_cols );

    u64 num_rows = src.num_rows;
    u64 num_cols = src.num_cols;

    i32 elem_size = sizeof(T);

    u64 lda = src.ld_rows;
    u64 ldb = dst.ld_rows;

    DRY_RETURN(src);

    cublasCheck( 
        cublasGetMatrixAsync(
            num_rows, num_cols,
            elem_size,
            src.ptr(), lda,
            dst.ptr(), ldb,
            stream
        )
    );
}

template <class T>
void 
set_matrix(
    HostTensor<T> &src,
    DeviceTensor<T> &dst,
    cudaStream_t stream
) {
    ASSERT( src.num_rows == dst.num_rows );
    ASSERT( src.num_cols == dst.num_cols );

    u64 num_rows = src.num_rows;
    u64 num_cols = src.num_cols;

    i32 elem_size = sizeof(T);

    u64 lda = src.ld_rows;
    u64 ldb = dst.ld_rows;

    DRY_RETURN(src);

    cublasCheck( 
        cublasSetMatrixAsync(
            num_rows, num_cols,
            elem_size,
            src.ptr(), lda,
            dst.ptr(), ldb,
            stream
        )
    );
}


template <class T>
void
tensor_set_zero(
    HostTensor<T> &tensor
) {
    DRY_RETURN(tensor);

    memset(tensor.ptr(), 0, tensor.ld_rows * tensor.num_cols * sizeof(T));
}

template <class T>
void
tensor_set_zero(
    DeviceTensor<T> &tensor,
    cudaStream_t stream
) {
    DRY_RETURN(tensor);

    cuCheck( cudaMemsetAsync(tensor.ptr(), 0, tensor.ld_rows * tensor.num_cols * sizeof(T), stream) );
}

template <class T>
void
tensor_copy_to(
	DeviceTensor<T> &src,
	DeviceTensor<T> &dst,
    cudaStream_t stream
) {
	u64 data_type_size = sizeof(T);

	ASSERT( dst.num_rows == src.num_rows );
	ASSERT( dst.num_cols == src.num_cols );

    if (dst.is_dry_run()) return;

    cuCheck(
        cudaMemcpy2DAsync(
            dst.ptr(), 
            dst.ld_rows * data_type_size,
            src.ptr(),
            src.ld_rows * data_type_size,
            dst.num_rows * data_type_size,
            dst.num_cols,
            cudaMemcpyDeviceToDevice,
            stream
        )
    );
}

static
void
tensor_print(
    HostStackAllocator &hsa, 
    HostTensor<f32> &h_tensor, 
    bool row_major
) {
    if (!row_major) {
        f32 *h_ptr = h_tensor.ptr();
        for (u64 row = 0; row < h_tensor.num_rows; row++) {
            for (u64 col = 0; col < h_tensor.num_cols; col++) {
                f32 f = h_ptr[col * h_tensor.ld_rows + row];
                printf("%g, ", f);
            }
            if (h_tensor.num_cols != 1) printf("\n");
        }
        if (h_tensor.num_cols == 1) printf("\n");
    } else {
        f32 *h_ptr = h_tensor.ptr();
        for (u64 col = 0; col < h_tensor.num_cols; col++) {
            for (u64 row = 0; row < h_tensor.num_rows; row++) {
                f32 f = h_ptr[col * h_tensor.ld_rows + row];
                printf("%g, ", f);
            }
            if (h_tensor.num_rows != 1) printf("\n");
        }
        if (h_tensor.num_rows == 1) printf("\n");
    }
}

static
void
tensor_print(
    HostStackAllocator &hsa, 
    DeviceTensor<f32> &tensor, 
    cudaStream_t stream,
    bool row_major = false
) {
    HostTensor<f32> h_tensor(hsa, tensor.num_rows, tensor.num_cols);
    get_matrix(tensor, h_tensor, stream);
    cuCheck( cudaStreamSynchronize(stream) );
    tensor_print(hsa, h_tensor, row_major);
}

template <class T>
static
const char *
_tensor_print_formatting_string() {
    static_assert(
        std::is_same_v<T, f32> || 
        std::is_same_v<T, f64> || 
        std::is_same_v<T,i32> || 
        std::is_same_v<T,u32> ||
        std::is_pointer<T>::value
    );

    if constexpr (std::is_same_v<T, f32>) {
        return "%f";
    } else if constexpr(std::is_same_v<T, f64>) {
        return "%f";
    } else if constexpr(std::is_same_v<T, i32>) {
        return "%d";
    } else if constexpr(std::is_same_v<T, u32>) {
        return "%u";
    } else if constexpr(std::is_same_v<T, u64>) {
        return "%llu";
    } else if constexpr(std::is_pointer<T>::value) {
        return "%p";
    } else {
        ASSERT( false );
    }
}

template <class T>
void
tensor_print_edge(
    HostStackAllocator &hsa,
	const char *str, 
	HostTensor<T> &tensor, 
	i32 edge_items
) {
    static_assert(
        std::is_same_v<T, f32> || 
        std::is_same_v<T, f64> || 
        std::is_same_v<T,i32> || 
        std::is_same_v<T,u32> ||
        std::is_pointer<T>::value
    );

    DRY_RETURN(hsa);
    
    u64 limit = edge_items;
	printf("%s (%zu, %zu)\n", str, tensor.num_rows, tensor.num_cols);
    T *h_ptr = tensor.ptr();
    for (i32 n = 0; n < tensor.num_rows; n++) {
        for (i32 d = 0; d < tensor.num_cols; d++) {
            T v = h_ptr[d * tensor.ld_rows + n];
            printf(_tensor_print_formatting_string<T>(), v);
            printf(", ");
            if (d == limit - 1 && tensor.num_cols > limit * 2) {
                printf(" ... ");
                d = tensor.num_cols - limit - 1;
            }
        }
        printf("\n");
        if (n == limit - 1 && tensor.num_rows > limit * 2) {
            printf("...,\n");
            n = tensor.num_rows - limit - 1;
        }
    }
}

template <class T>
void
tensor_print_edge(
    HostStackAllocator &hsa,
	const char *str, 
	DeviceTensor<T> &tensor, 
	i32 edge_items,
    cudaStream_t stream
) {
    HostTensor<T> h_tensor(hsa, tensor.num_rows, tensor.num_cols);
    get_matrix(tensor, h_tensor, stream);
    cuCheck( cudaStreamSynchronize(stream) );
    tensor_print_edge(hsa, str, h_tensor, edge_items);
}

}
