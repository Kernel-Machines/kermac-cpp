#pragma once

#include <kermac.cuh>
#include <philox.cuh>

namespace kermac {

enum class RNGType {
    UNIFORM,
    NORMAL
};

struct Philox {
    DeviceConstant<u32> state_0;
    DeviceConstant<u32> state_1;
    u64 seed;
    bool swap;

    Philox(
        DeviceStackAllocator &dsa,
        u64 seed,
        cudaStream_t stream
    ) : state_0(dsa, (u32)0, stream),
        state_1(dsa, (u32)0, stream),
        seed(seed),
        swap(false)
    {}

    // Gets the right state and then swaps the state variables for next time.
    void 
    get(
        u32 **state_0_ptr, 
        u32 **state_1_ptr
    ) {
        *state_0_ptr = swap ? state_0.ptr() : state_1.ptr();
        *state_1_ptr = swap ? state_1.ptr() : state_0.ptr();
        swap = !swap;
    }
    
    ~Philox() {}

    Philox(const Philox&) = delete;
    Philox& operator=(const Philox&) = delete;

    Philox(Philox&&) = delete;
    Philox& operator=(Philox&&) = delete;
};


template<RNGType rng_type, class T>
__forceinline__
__device__
auto 
philox4_template(
	u32 thread_offset, 
	u32 seed, 
	u32 *iter_count
) {
	static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
	
	if constexpr (rng_type == RNGType::NORMAL && std::is_same_v<T, f32>) {
		return philox_normal4(thread_offset, seed, iter_count);
	} else if constexpr (rng_type == RNGType::UNIFORM && std::is_same_v<T, f32>) {
		return philox_uniform4(thread_offset, seed, iter_count);
	} else if constexpr (rng_type == RNGType::NORMAL && std::is_same_v<T, f64>) {
		return philox_normal4_double(thread_offset, seed, iter_count);
	} else if constexpr (rng_type == RNGType::UNIFORM && std::is_same_v<T, f64>) {
		return philox_uniform4_double(thread_offset, seed, iter_count);
	}
}

template <RNGType rng_type, class T>
static
__global__
void
kernel_rng_2D(
	const u64 num_rows,
	const u64 ld_rows,
	const u64 num_cols,
	const u32 seed,
	const T scale,
	const T shift,
	u32 * __restrict__ philox_state,
	u32 * __restrict__ philox_max_state,
	T * __restrict__ data
) {
	static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

	u64 row = threadIdx.x + blockIdx.x * blockDim.x;
    u64 col = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < num_cols) {
		u64 blockId = blockIdx.x + blockIdx.y * gridDim.x;
		u64 threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

		const u64 rng_idx = threadId;
		u32 local_state = *philox_state;
		auto rng = philox4_template<rng_type,T>(rng_idx, seed, &local_state);
		for (u64 i = 0; i < 4; i++) {
			if (row < num_rows) {
				data[col * ld_rows + row] = scale * reinterpret_cast<T*>(&rng)[i] + shift;
				row += gridDim.x * blockDim.x;
			}
		}
		philox_put_state(local_state, philox_max_state);
	}
}

template <class T>
void
tensor_rng(
	Philox &philox,
	RNGType rng_type,
	DeviceTensor<T> &a,
	T scale, T shift,
	cudaStream_t stream
) {
	static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

	u64 num_rows = a.num_rows;
	u64 ld_rows = a.ld_rows;
	u64 num_cols = a.num_cols;

	int effective_rows = NEAREST_LARGER_MULTIPLE(num_rows, 4);

	dim3 block(32, 8);
	dim3 grid(
		NEAREST_LARGER_MULTIPLE(effective_rows, block.x),
		NEAREST_LARGER_MULTIPLE(num_cols, block.y)
	);

	u32 *state_0;
	u32 *state_1;

	DRY_RETURN(a);

	philox.get(&state_0, &state_1);

	if (rng_type == RNGType::UNIFORM) {
		kernel_rng_2D<RNGType::UNIFORM><<<grid, block, 0, stream>>>(
			num_rows, ld_rows, num_cols,
			philox.seed, 
			scale, shift,
			state_0,
			state_1,
			a.ptr()
		);
	} else if (rng_type == RNGType::NORMAL) {
		kernel_rng_2D<RNGType::NORMAL><<<grid, block, 0, stream>>>(
			num_rows, ld_rows, num_cols,
			philox.seed, 
			scale, shift,
			state_0,
			state_1,
			a.ptr()
		);
	}
}

}
