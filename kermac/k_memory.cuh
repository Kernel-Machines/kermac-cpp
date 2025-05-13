#pragma once

#include <kermac.cuh>

namespace kermac {

static constexpr u64 HOST_ALLOC_ALIGNMENT = 64ull;
static constexpr u64 DEVICE_ALLOC_ALIGNMENT = 512ull;

static constexpr u64 CONSTANT_ALIGNMENT_BYTES = 16ull;

enum class MemorySpace {
    Host,
    Device
};

enum class MemoryType {
    FULL,
    VIEW
};

template <MemorySpace memory_space>
struct StackAllocator {
    u8 *memory_ptr;
    u64 current_offset;
    u64 current_stack_counter;
    u64 largest_total_offset;
    u64 allocated_bytes;
    bool dry_run;

    // Stream allocation handling
    u64 stream_snapshot_offset = 0;
    u64 stream_total_offset = 0;
    bool is_snapshot = false;
    bool is_stream_snapshot = false;

    StackAllocator(
        u64 total_num_bytes
    ) : current_offset(0), 
        current_stack_counter(0),
        largest_total_offset(0),
        allocated_bytes(total_num_bytes),
        dry_run(false)
    {
        if constexpr (memory_space == MemorySpace::Host) {
            void *ptr;
            ASSERT( posix_memalign(&ptr, HOST_ALLOC_ALIGNMENT, total_num_bytes) == 0 );
            memory_ptr = static_cast<u8*>(ptr);
        } else if constexpr (memory_space == MemorySpace::Device) {
            cuCheck( cudaMalloc(&memory_ptr, total_num_bytes) );
        }
    }

    StackAllocator(
    ) : current_offset(0),
        current_stack_counter(0),
        largest_total_offset(0),
        allocated_bytes(0),
        dry_run(true) 
    {}

    ~StackAllocator() {
        if (!dry_run) {
            if constexpr (memory_space == MemorySpace::Host) {
                free(memory_ptr);
            } else if constexpr (memory_space == MemorySpace::Device) {
                cuCheck( cudaFree(memory_ptr) );
            }
        }
    }

    // Prevent copying.
    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;

    // Prevent moving.
    StackAllocator(StackAllocator&&) = delete;
    StackAllocator& operator=(StackAllocator&&) = delete;

    // Record the current offset before all the streams
    void
    stream_snapshot() {
        ASSERT( !is_snapshot );
        ASSERT( !is_stream_snapshot );
        ASSERT( stream_snapshot_offset == 0 );
        ASSERT( stream_total_offset == 0 );

        stream_snapshot_offset = current_offset;
        is_snapshot = true;
    }

    // Replace the largest offset with the current offset
    // to measure the increment in the functions that follow
    void
    stream_alloc_start() {
        ASSERT( is_snapshot );
        ASSERT( !is_stream_snapshot );
        ASSERT( stream_total_offset == 0 );

        stream_total_offset = largest_total_offset;
        largest_total_offset = current_offset;
        is_stream_snapshot = true;
    }

    // Measure the largest offset to know how much memory the stream allocated
    // at its maximum. Use this to offset the pointer so the next stream won't clobber
    // this memory.
    // Update the largest_total_offset to be globally correct
    void
    stream_alloc_end() {
        ASSERT( is_snapshot );
        ASSERT( is_stream_snapshot );
        
        u64 stream_bytes_allocated = largest_total_offset - current_offset;
        largest_total_offset = MAX(stream_total_offset, current_offset + stream_bytes_allocated);
        current_offset = current_offset + stream_bytes_allocated;

        stream_total_offset = 0;
        is_stream_snapshot = false;
    }

    // Restore all of the memory back to the snapshotted state.
    // Use this with cudaDeviceSynchronize() before the main stream allocates again
    void
    stream_restore() {
        ASSERT( is_snapshot );
        ASSERT( !is_stream_snapshot );

        current_offset = stream_snapshot_offset;
        stream_snapshot_offset = 0;
        stream_total_offset = 0;
        is_snapshot = false;
    }

    bool
    is_dry_run() {
        return dry_run;
    }

    void
    clear() {
        if (!dry_run) {
            if constexpr (memory_space == MemorySpace::Host) {
                free(memory_ptr);
            } else if constexpr (memory_space == MemorySpace::Device) {
                cuCheck( cudaFree(memory_ptr) );
            }
        }

        current_offset = 0;
        current_stack_counter = 0;
        allocated_bytes = largest_total_offset;
        largest_total_offset = 0;

        dry_run = true;
    }

    void
    alloc() {
        ASSERT( dry_run );
        dry_run = false;

        if constexpr (memory_space == MemorySpace::Host) {
            printf("host_bytes: %" PRIu64 "\n", largest_total_offset);
            void *ptr;
            ASSERT( posix_memalign(&ptr, HOST_ALLOC_ALIGNMENT, largest_total_offset) == 0 );
            memory_ptr = static_cast<u8*>(ptr);
        } else if constexpr (memory_space == MemorySpace::Device) {
            printf("device_bytes: %" PRIu64 "\n", largest_total_offset);
            cuCheck( cudaMalloc(&memory_ptr, largest_total_offset) );
        }

        current_offset = 0;
        current_stack_counter = 0;
        allocated_bytes = largest_total_offset;
        largest_total_offset = 0;
    }
};

using DeviceStackAllocator = StackAllocator<MemorySpace::Device>;
using HostStackAllocator = StackAllocator<MemorySpace::Host>;

template <MemorySpace memory_space>
static
u64
_memory_calc_default_alignment() {
    if constexpr (memory_space == MemorySpace::Host) {
        return HOST_ALLOC_ALIGNMENT;
    } else if constexpr (memory_space == MemorySpace::Device) {
        return DEVICE_ALLOC_ALIGNMENT;
    } else {
        ASSERT( false );
    }
}

template <MemorySpace memory_space>
struct Memory {
    u64 offset;
    StackAllocator<memory_space> &stack_allocator;
    const u64 stack_count;
    const u64 num_bytes;
    const MemoryType memory_type;
    Memory(
        StackAllocator<memory_space> &sa,
        u64 num_bytes,
        u64 alignment_bytes
    ) : stack_allocator(sa),
        stack_count(sa.current_stack_counter++),
        num_bytes(num_bytes),
        memory_type(MemoryType::FULL)
    {
        // If its zero make sure it jumps at least a little.
        if (num_bytes == 0) num_bytes = 4;
        sa.current_offset = alignment_bytes * NEAREST_LARGER_MULTIPLE(sa.current_offset, alignment_bytes);
    
        if (!sa.is_dry_run()) {
            ASSERT( sa.current_offset + num_bytes <= sa.allocated_bytes );
        }
    
        offset = sa.current_offset;
    
        sa.current_offset += num_bytes;
        sa.largest_total_offset = MAX(sa.largest_total_offset, sa.current_offset);
    }

    Memory(
        StackAllocator<memory_space> &sa,
        u64 num_bytes
    ) : Memory(sa, num_bytes, _memory_calc_default_alignment<memory_space>())
    {}

    Memory(
        Memory<memory_space> &memory, 
        u64 view_offset
    ) : stack_allocator(memory.stack_allocator),
        stack_count(stack_allocator.current_stack_counter++),
        memory_type(MemoryType::VIEW),
        offset(memory.offset + view_offset),
        num_bytes(memory.num_bytes - view_offset)
    {}

    ~Memory() {
        // Check to make sure that there are no out of order frees
        ASSERT( offset < stack_allocator.current_offset );
    
        // Make sure that the tagged memory has the expected stack depth
        stack_allocator.current_stack_counter--;
        ASSERT( stack_count == stack_allocator.current_stack_counter );
    
        // If its a view make sure the stack counter is sane, but don't destroy the memory
        if (memory_type == MemoryType::VIEW) return;
    
        stack_allocator.current_offset = offset;
        // Null out stack allocator so it will crash on reuse
    }

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    Memory(Memory&&) noexcept = delete;
    Memory& operator=(Memory&&) noexcept = delete;

    bool
    is_dry_run() {
        return stack_allocator.dry_run;
    }

    u8 *ptr() {
        ASSERT( !is_dry_run() );
        return stack_allocator.memory_ptr + offset;
    }
};

using DeviceMemory = Memory<MemorySpace::Device>;
using HostMemory = Memory<MemorySpace::Host>;

template <class T>
struct DeviceConstant {
    Memory<MemorySpace::Device> device_memory;
    DeviceConstant(
        StackAllocator<MemorySpace::Device> &dsa,
        T t, 
        cudaStream_t stream
    ) : device_memory(DeviceMemory(dsa, sizeof(T), CONSTANT_ALIGNMENT_BYTES)) 
    {
        set(t, stream);
    }

    DeviceConstant(
        StackAllocator<MemorySpace::Device> &dsa
    ) : device_memory(DeviceMemory(dsa, sizeof(T), CONSTANT_ALIGNMENT_BYTES)) 
    {}

    DeviceConstant(
        Memory<MemorySpace::Device> &device_memory,
        u64 elems_offset
    ) : device_memory(device_memory, sizeof(T) * elems_offset)
    {}

    ~DeviceConstant() {}

    DeviceConstant(const DeviceConstant&) = delete;
    DeviceConstant& operator=(const DeviceConstant&) = delete;

    DeviceConstant(DeviceConstant&&) = delete;
    DeviceConstant& operator=(DeviceConstant&&) = delete;

    T get(
        cudaStream_t stream
    ) {
        T t;
        cuCheck( cudaMemcpyAsync(&t, device_memory.ptr(), sizeof(T), cudaMemcpyDeviceToHost, stream) );
        cuCheck( cudaStreamSynchronize(stream) );
        return t;
    }

    void 
    set(
        T t, 
        cudaStream_t stream
    ){
        if (device_memory.is_dry_run()) return;

        cuCheck( cudaMemcpyAsync(device_memory.ptr(), &t, sizeof(T), cudaMemcpyHostToDevice, stream) );
    }

    bool
    is_dry_run() {
        return device_memory.is_dry_run();
    }

    T *ptr() {
        return reinterpret_cast<T*>(device_memory.ptr());
    }
};

template <class T>
struct DeviceConstants {
    DeviceConstant<T> one;
    DeviceConstant<T> zero;
    DeviceConstant<T> negative_one;
    DeviceConstants(
        StackAllocator<MemorySpace::Device> &dsa,
        cudaStream_t stream
    ) : one(dsa, c_one<T>, stream),
        zero(dsa, c_zero<T>, stream),
        negative_one(dsa, c_negative_one<T>, stream) 
    {}

    ~DeviceConstants() {}

    DeviceConstants(const DeviceConstants&) = delete;
    DeviceConstants& operator=(const DeviceConstants&) = delete;

    DeviceConstants(DeviceConstants&&) = delete;
    DeviceConstants& operator=(DeviceConstants&&) = delete;
};

}
