#pragma once

#include <kermac.cuh>
#include <cusolverDn.h>

#define cusolverCheck(ans) checkCusolverStatus((ans), __FILE__, __LINE__)
static void checkCusolverStatus(cusolverStatus_t status, const char *file, int line) {
	if (status != CUSOLVER_STATUS_SUCCESS) {
		printf("cuSolver API failed with status %d %s %d\n", status, file, line);
		exit(1);
	}
}

namespace kermac {

struct CUSolver {
    cusolverDnHandle_t handle;
    CUSolver(){
        cusolverCheck( cusolverDnCreate(&handle) );
    }

    ~CUSolver(){
        cusolverCheck( cusolverDnDestroy(handle) );
    }

    CUSolver(const CUSolver&) = delete;
    CUSolver& operator=(const CUSolver&) = delete;

    CUSolver(CUSolver&&) = delete;
    CUSolver& operator=(CUSolver&&) = delete;
};

static
bool
cusolver_check_infos(
	HostStackAllocator &hsa,
	DeviceMultiTensor<i32> &infos,
	cudaStream_t stream
) {
	ASSERT( infos.num_modes == 1 );

	u64 B = infos.extent[0];
	HostMultiTensor<i32> h_infos(hsa, B);
	get_matrix(infos.tensor, h_infos.tensor, stream);
	cuCheck( cudaStreamSynchronize(stream) );

	DRY_COND(hsa) return true;

	for (i32 b = 0; b < B; b++) {
		i32 *h_info_ptr = h_infos.tensor.ptr();
		if (h_info_ptr[b] != 0) {
			return false;
		}
	}
	return true;
}

template <class T>
void
cusolver_potrf(
    CUSolver &cusolver,
	DeviceStackAllocator &dsa,
    cublasFillMode_t uplo,
    DeviceConstant<i32> &info,
    DeviceTensor<T> &a,
	cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    cusolverDnParams_t params = NULL;

    cudaDataType data_type = std::is_same_v<T,f32> ? CUDA_R_32F : CUDA_R_64F;
	cudaDataType data_type_a = data_type;
	cudaDataType compute_type = data_type;

    i64 N = a.num_rows;
	i64 stride_a = a.ld_rows;

    ASSERT( N == a.num_cols );

	u64 workspace_bytes = 0;
	u64 workspace_bytes_host = 0;

	cusolverCheck(
		cusolverDnXpotrf_bufferSize(
			cusolver.handle,
			params,
			uplo,
			N,
			data_type_a,
			(void*)NULL,
			stride_a,
			compute_type,
			&workspace_bytes,
			&workspace_bytes_host
		)
	);

	// I've never seen this anything but 0
	ASSERT( workspace_bytes_host == 0 );
	DeviceMemory workspace(dsa, workspace_bytes);

	DRY_RETURN(a);

	cudaStream_t prev_stream;
	cusolverCheck( cusolverDnGetStream(cusolver.handle, &prev_stream) );
	cusolverCheck( cusolverDnSetStream(cusolver.handle, stream) );

	cusolverCheck(
		cusolverDnXpotrf(
			cusolver.handle,
			params,
			uplo,
			N,
			data_type_a,
			a.ptr(),
			stride_a,
			compute_type,
			workspace.ptr(),
			workspace_bytes,
			NULL,
			workspace_bytes_host,
			info.ptr()
		)
	);
	
	cusolverCheck( cusolverDnSetStream(cusolver.handle, prev_stream) );
}

template <class T>
void
cusolver_potrs(
    CUSolver &cusolver,
    cublasFillMode_t uplo,
    DeviceConstant<i32> &info,
    DeviceTensor<T> &a,
    DeviceTensor<T> &b,
	cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

    cusolverDnParams_t params = NULL;

    cudaDataType data_type = std::is_same_v<T,f32> ? CUDA_R_32F : CUDA_R_64F;
	cudaDataType data_type_a = data_type;
	cudaDataType data_type_b = data_type;

	i64 N = a.num_rows;
	i64 nrhs = b.num_cols;
	i64 ldA = a.ld_rows;
	i64 ldB = b.ld_rows;

    ASSERT( N == a.num_cols );
	ASSERT( N == b.num_rows );

	DRY_RETURN(a);

	cudaStream_t prev_stream;
	cusolverCheck( cusolverDnGetStream(cusolver.handle, &prev_stream) );
	cusolverCheck( cusolverDnSetStream(cusolver.handle, stream) );

	cusolverCheck(
		cusolverDnXpotrs(
			cusolver.handle,
			params,
			uplo,
			N,
			nrhs,
			data_type_a,
			a.ptr(),
			ldA,
			data_type_b,
			b.ptr(),
			ldB,
			info.ptr()
		)
	);

	cusolverCheck( cusolverDnSetStream(cusolver.handle, prev_stream) );
}

template <class T>
void
cusolver_syevdx(
	CUSolver &cusolver,
    HostStackAllocator &hsa,
    DeviceStackAllocator &dsa,
	cublasFillMode_t uplo,
	DeviceConstant<i32> &info,
	DeviceTensor<T> &a,
	DeviceTensor<T> &w,
	i32 sub_start,
	i32 sub_end,
	cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

	cusolverDnParams_t params = NULL;
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

	cudaDataType data_type = std::is_same_v<T,f32> ? CUDA_R_32F : CUDA_R_64F;
	cudaDataType data_type_a = data_type;
	cudaDataType data_type_w = data_type;
	cudaDataType compute_type = data_type;

	i64 N = a.num_rows;
	i64 ldN = a.ld_rows;

	ASSERT( N == a.num_cols );
	ASSERT( w.num_rows == N );
	ASSERT( w.num_cols == 1 );

	f32 vl = 0.0f;
	f32 vu = 0.0f;

	i64 il = sub_start;
	i64 iu = sub_end;
	i64 h_meig;

	u64 workspace_in_bytes_on_device;
	u64 workspace_in_bytes_on_host;

	cusolverCheck(
		cusolverDnXsyevdx_bufferSize(
			cusolver.handle,
			params,
			jobz,
			range,
			uplo,
			N,
			data_type_a,
			NULL,
			ldN,
			&vl,
			&vu,
			il,
			iu,
			&h_meig,
			data_type_w,
			NULL,
			compute_type,
			&workspace_in_bytes_on_device,
			&workspace_in_bytes_on_host
		)
	);

	DeviceMemory workspace_device(dsa, workspace_in_bytes_on_device, DEVICE_ALLOC_ALIGNMENT);
    HostMemory workspace_host(hsa, workspace_in_bytes_on_host, HOST_ALLOC_ALIGNMENT);

	DRY_RETURN(a);

	cudaStream_t prev_stream;
	cusolverCheck( cusolverDnGetStream(cusolver.handle, &prev_stream) );
	cusolverCheck( cusolverDnSetStream(cusolver.handle, stream) );

	cusolverCheck(
		cusolverDnXsyevdx(
			cusolver.handle,
			params,
			jobz,
			range,
			uplo,
			N,
			data_type_a,
			a.ptr(),
			ldN,
			&vl,
			&vu,
			il,
			iu,
			&h_meig,
			data_type_w,
			w.ptr(),
			compute_type,
			workspace_device.ptr(),
			workspace_in_bytes_on_device,
			workspace_host.ptr(),
			workspace_in_bytes_on_host,
			info.ptr()
		)
	);

	cusolverCheck( cusolverDnSetStream(cusolver.handle, prev_stream) );
}

template <class T>
void
cusolver_syevd(
	CUSolver &cusolver,
	HostStackAllocator &hsa,
	DeviceStackAllocator &dsa,
	cublasFillMode_t uplo,
	DeviceConstant<i32> &info,
	DeviceTensor<T> &a,
	DeviceTensor<T> &w,
	cudaStream_t stream
) {
	static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

	cusolverDnParams_t params = NULL;
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

	cudaDataType data_type = std::is_same_v<T,f32> ? CUDA_R_32F : CUDA_R_64F;
	cudaDataType data_type_a = data_type;
	cudaDataType data_type_w = data_type;
	cudaDataType compute_type = data_type;

	i64 N = a.num_rows;
	i64 ldN = a.ld_rows;

	ASSERT( N == a.num_rows );
	ASSERT( N == a.num_cols );
	ASSERT( N == w.num_rows );
	ASSERT( 1 == w.num_cols );
	
	u64 workspace_in_bytes_on_device;
	u64 workspace_in_bytes_on_host;

	cusolverCheck(
		cusolverDnXsyevd_bufferSize(
			cusolver.handle,
			params,
			jobz,
			uplo,
			N,
			data_type_a,
			NULL,
			ldN,
			data_type_w,
			NULL,
			compute_type,
			&workspace_in_bytes_on_device,
			&workspace_in_bytes_on_host
		)
	);

	DeviceMemory workspace_device(dsa, workspace_in_bytes_on_device, DEVICE_ALLOC_ALIGNMENT);
    HostMemory workspace_host(hsa, workspace_in_bytes_on_host, HOST_ALLOC_ALIGNMENT);

	DRY_RETURN(a);
	
	cudaStream_t prev_stream;
	cusolverCheck( cusolverDnGetStream(cusolver.handle, &prev_stream) );
	cusolverCheck( cusolverDnSetStream(cusolver.handle, stream) );

	cusolverCheck(
		cusolverDnXsyevd(
			cusolver.handle,
			params,
			jobz,
			uplo,
			N,
			data_type_a,
			a.ptr(),
			ldN,
			data_type_w,
			w.ptr(),
			compute_type,
			workspace_device.ptr(),
			workspace_in_bytes_on_device,
			workspace_host.ptr(),
			workspace_in_bytes_on_host,
			info.ptr()
		)
	);

	cusolverCheck( cusolverDnSetStream(cusolver.handle, prev_stream) );
}

// Turning this off because it's useless until nvidia improves it.
#if 0
template <class T>
void
cusolver_syevd_batched(
	CUSolver &cusolver,
	HostStackAllocator &hsa,
	DeviceStackAllocator &dsa,
	cublasFillMode_t uplo,
	DeviceMultiTensor<i32> &infos, // B
	DeviceMultiTensor<T> &a, // N,N,B
	DeviceMultiTensor<T> &w, // N, B
	cudaStream_t stream
) {
	static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
	ASSERT( infos.num_modes == 1 );
	ASSERT( a.num_modes == 3 );
	ASSERT( w.num_modes == 2 );

	i64 N = a.extent[0];
	i64 B = a.extent[2];
	
	ASSERT( N == a.extent[0] );
	ASSERT( N == a.extent[1] );
	ASSERT( N == w.extent[0] );

	ASSERT( B == infos.extent[0] );
	ASSERT( B == a.extent[2] );
	ASSERT( B == w.extent[1] );

	cusolverDnParams_t params = NULL;
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

	cudaDataType data_type = std::is_same_v<T,f32> ? CUDA_R_32F : CUDA_R_64F;
	cudaDataType data_type_a = data_type;
	cudaDataType data_type_w = data_type;
	cudaDataType compute_type = data_type;
	
	u64 workspace_in_bytes_on_device;
	u64 workspace_in_bytes_on_host;

	i64 ldN = a.tensor.ld_rows;

	cusolverCheck(
		cusolverDnXsyevBatched_bufferSize(
			cusolver.handle,
			params,
			jobz, 
			uplo,
			N, data_type_a, NULL,
			ldN,
			data_type_w,
			NULL,
			compute_type,
			&workspace_in_bytes_on_device,
			&workspace_in_bytes_on_host,
			B
		)
	);

	DeviceMemory workspace_device(dsa, workspace_in_bytes_on_device, DEVICE_ALLOC_ALIGNMENT);
    HostMemory workspace_host(hsa, workspace_in_bytes_on_host, HOST_ALLOC_ALIGNMENT);

	DeviceMultiTensor<T> w_packed_stride(dsa, N * B);

	DRY_RETURN(a);

	cudaStream_t prev_stream;
	cusolverCheck( cusolverDnGetStream(cusolver.handle, &prev_stream) );
	cusolverCheck( cusolverDnSetStream(cusolver.handle, stream) );

	cusolverCheck(
		cusolverDnXsyevBatched(
			cusolver.handle,
			params,
			jobz,
			uplo,
			N,
			data_type_a,
			a.tensor.ptr(),
			ldN,
			data_type_w,
			w_packed_stride.tensor.ptr(),
			compute_type,
			workspace_device.ptr(),
			workspace_in_bytes_on_device,
			workspace_host.ptr(),
			workspace_in_bytes_on_host,
			infos.tensor.ptr(),
			B
		)
	);

	cusolverCheck( cusolverDnSetStream(cusolver.handle, prev_stream) );

	// batched_syev function doesnt support a leading dimension on the eigenvalues..
	// Need to copy it to the proper alignment
	u64 data_type_size = sizeof(T);
	cuCheck(
        cudaMemcpy2DAsync(
            w.tensor.ptr(), 
            w.tensor.ld_rows * data_type_size,
            w_packed_stride.tensor.ptr(),
            N * data_type_size,
            N * data_type_size,
            B,
            cudaMemcpyDeviceToDevice,
			stream
        )
    );
}
#endif

}
