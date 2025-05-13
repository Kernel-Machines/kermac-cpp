#pragma once

#include <kermac.cuh>
#include <cutensor.h>

#define cutensorCheck(ans) checkCutensorStatus((ans), __FILE__, __LINE__)
static void checkCutensorStatus(cutensorStatus_t status, const char *file, int line) {
	if (status != CUTENSOR_STATUS_SUCCESS) {
		const char* error_string = cutensorGetErrorString(status);
		printf("cuTENSOR API failed with status %d %s %s %d\n", status, error_string, file, line);
		exit(1);
	}
}

namespace kermac {

enum class TensorCoreMode {
    FP32,       // Standard FP32
    FP64,       // Standard FP64
    TF32,       // TF32 Tensor Core
    TF32_3X     // 3xTF32 Improved Accuracy
};

static const cutensorJitMode_t JIT_MODE = CUTENSOR_JIT_MODE_NONE;

struct CUTensor {
    cutensorHandle_t handle;
    cudaStream_t current_stream;
    CUTensor() : current_stream(NULL) {
        CPUTimer timer;
        cutensorCheck( cutensorCreate(&handle) );
        f32 seconds = timer.seconds();
        printf("Initialized cuTensor in %f seconds\n", seconds);
    }

    ~CUTensor() {
        cutensorCheck( cutensorDestroy(handle) );
    }

    CUTensor(const CUTensor&) = delete;
    CUTensor& operator=(const CUTensor&) = delete;

    CUTensor(CUTensor&&) = delete;
    CUTensor& operator=(CUTensor&&) = delete;
};

template <class T>
static
cutensorTensorDescriptor_t
cutensor_tensor_descriptor_create(
    CUTensor &cutensor,
    DeviceMultiTensor<T> &a
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>, "Unsupported floating type");

    cutensorDataType_t cutensor_type_a;
    if constexpr (std::is_same_v<T, f32>) {
        cutensor_type_a = CUTENSOR_R_32F;
    } else if constexpr (std::is_same_v<T, f64>) {
        cutensor_type_a = CUTENSOR_R_64F;
    }

    cutensorTensorDescriptor_t cutensor_desc_a;
    cutensorCheck(
        cutensorCreateTensorDescriptor(
            cutensor.handle,
            &cutensor_desc_a,
            a.num_modes,
            a.extent,
            a.stride,
            cutensor_type_a,
            DEVICE_ALLOC_ALIGNMENT
        )
    );

    return cutensor_desc_a;
}

template <class T>
static
void
cutensor_contraction(
    CUTensor &cutensor,
    DeviceStackAllocator &dsa,  
    TensorCoreMode tcm,
    T alpha,
    DeviceMultiTensor<T> &a, const char *modes_a,
    DeviceMultiTensor<T> &b, const char *modes_b,
    T beta,
    DeviceMultiTensor<T> &c, const char *modes_c,
    DeviceMultiTensor<T> &d, const char *modes_d,
    cudaStream_t stream,
    f32 *flops_ref = NULL
) {
    ASSERT( strlen(modes_a) == a.num_modes );
    ASSERT( strlen(modes_b) == b.num_modes );
    ASSERT( strlen(modes_c) == c.num_modes );
    ASSERT( strlen(modes_d) == d.num_modes );

    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>, "Unsupported floating type");

    cutensorComputeDescriptor_t cutensor_desc_compute;
    if constexpr (std::is_same_v<T, f32>) {
        switch(tcm) {
            case TensorCoreMode::FP32: {
                cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_32F;
            } break;
            case TensorCoreMode::TF32: {
                cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_TF32;
            } break;
            case TensorCoreMode::TF32_3X: {
                cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_3XTF32;
            }
            default:
                ASSERT( false );
        }
    } else if constexpr (std::is_same_v<T, f64>) {
        ASSERT( tcm == TensorCoreMode::FP64 );
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_64F;
    }

    i32 i_modes_a[4] = {0};
    i32 i_modes_b[4] = {0};
    i32 i_modes_c[4] = {0};
    i32 i_modes_d[4] = {0};

    for (i32 i = 0; i < a.num_modes; i++) i_modes_a[i] = (i32)modes_a[i];
    for (i32 i = 0; i < b.num_modes; i++) i_modes_b[i] = (i32)modes_b[i];
    for (i32 i = 0; i < c.num_modes; i++) i_modes_c[i] = (i32)modes_c[i];
    for (i32 i = 0; i < d.num_modes; i++) i_modes_d[i] = (i32)modes_d[i];

    cutensorTensorDescriptor_t cutensor_desc_a = cutensor_tensor_descriptor_create(cutensor, a);
    cutensorTensorDescriptor_t cutensor_desc_b = cutensor_tensor_descriptor_create(cutensor, b);
    cutensorTensorDescriptor_t cutensor_desc_c = cutensor_tensor_descriptor_create(cutensor, c);
    cutensorTensorDescriptor_t cutensor_desc_d = cutensor_tensor_descriptor_create(cutensor, d);

    cutensorOperationDescriptor_t cutensor_desc;
    cutensorCheck(
        cutensorCreateContraction(
            cutensor.handle,
            &cutensor_desc,
            cutensor_desc_a, i_modes_a, CUTENSOR_OP_IDENTITY,
            cutensor_desc_b, i_modes_b, CUTENSOR_OP_IDENTITY,
            cutensor_desc_c, i_modes_c, CUTENSOR_OP_IDENTITY,
            cutensor_desc_d, i_modes_d,
            cutensor_desc_compute
        )
    );

    cutensorDataType_t scalar_type;
    cutensorCheck(
        cutensorOperationDescriptorGetAttribute(
            cutensor.handle,
            cutensor_desc,
            CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
            (void*)&scalar_type,
            sizeof(scalar_type)
        )
    );

    if constexpr (std::is_same_v<T, f32>) {
        ASSERT( scalar_type == CUTENSOR_R_32F );
    } else if constexpr (std::is_same_v<T, f64>) {
        ASSERT( scalar_type == CUTENSOR_R_64F ); 
    }

    const cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t cutensor_plan_pref;
    cutensorCheck(
        cutensorCreatePlanPreference(
            cutensor.handle,
            &cutensor_plan_pref,
            cutensor_algo,
            JIT_MODE
        )
    );

    u64 cutensor_workspace_size_estimate = 0;
    const cutensorWorksizePreference_t cutensor_workspace_pref = CUTENSOR_WORKSPACE_DEFAULT;
    cutensorCheck(
        cutensorEstimateWorkspaceSize(
            cutensor.handle,
            cutensor_desc,
            cutensor_plan_pref,
            cutensor_workspace_pref,
            &cutensor_workspace_size_estimate
        )
    );

    cutensorPlan_t cutensor_plan;
    cutensorCheck(
        cutensorCreatePlan(
            cutensor.handle,
            &cutensor_plan,
            cutensor_desc,
            cutensor_plan_pref,
            cutensor_workspace_size_estimate
        )
    );

    u64 cutensor_actual_workspace_size = 0;
    cutensorCheck(
        cutensorPlanGetAttribute(
            cutensor.handle,
            cutensor_plan,
            CUTENSOR_PLAN_REQUIRED_WORKSPACE,
            &cutensor_actual_workspace_size,
            sizeof(cutensor_actual_workspace_size)
        )
    );

    DeviceMemory cutensor_workspace(dsa, cutensor_actual_workspace_size, DEVICE_ALLOC_ALIGNMENT);
    
    if (flops_ref != NULL) {
        f32 flops;
        cutensorCheck(
            cutensorOperationDescriptorGetAttribute(
                cutensor.handle,
                cutensor_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
                (void*)&flops,
                sizeof(flops)
            )
        );
        *flops_ref = flops;
    }

    WET_COND(dsa) {
        cutensorCheck(
            cutensorContract(
                cutensor.handle,
                cutensor_plan,
                (void*) &alpha, 
                a.tensor.ptr(), 
                b.tensor.ptr(),
                (void*) &beta, 
                c.tensor.ptr(), 
                d.tensor.ptr(),
                cutensor_workspace.ptr(),
                cutensor_actual_workspace_size,
                stream
            )
        );
    }

    cutensorCheck( cutensorDestroyPlan(cutensor_plan) );
    cutensorCheck( cutensorDestroyPlanPreference(cutensor_plan_pref) );
    cutensorCheck( cutensorDestroyOperationDescriptor(cutensor_desc) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_d) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_c) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_b) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_a) );
}

template <class T>
static
void
cutensor_contraction_trinary(
    CUTensor &cutensor,
    DeviceStackAllocator &dsa,  
    TensorCoreMode tcm,
    T alpha,
    DeviceMultiTensor<T> &a, const char *modes_a,
    DeviceMultiTensor<T> &b, const char *modes_b,
    DeviceMultiTensor<T> &c, const char *modes_c,
    T beta,
    DeviceMultiTensor<T> &d, const char *modes_d,
    DeviceMultiTensor<T> &e, const char *modes_e,
    cudaStream_t stream,
    f32 *flops_ref = NULL
) {
    ASSERT( strlen(modes_a) == a.num_modes );
    ASSERT( strlen(modes_b) == b.num_modes );
    ASSERT( strlen(modes_c) == c.num_modes );
    ASSERT( strlen(modes_d) == d.num_modes );
    ASSERT( strlen(modes_e) == e.num_modes );

    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>, "Unsupported floating type");

    cutensorComputeDescriptor_t cutensor_desc_compute;
    if constexpr (std::is_same_v<T, f32>) {
        switch(tcm) {
            case TensorCoreMode::FP32: {
                cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_32F;
            } break;
            case TensorCoreMode::TF32: {
                cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_TF32;
            } break;
            case TensorCoreMode::TF32_3X: {
                cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_3XTF32;
            }
            default:
                ASSERT( false );
        }
    } else if constexpr (std::is_same_v<T, f64>) {
        ASSERT( tcm == TensorCoreMode::FP64 );
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_64F;
    }

    i32 i_modes_a[4] = {0};
    i32 i_modes_b[4] = {0};
    i32 i_modes_c[4] = {0};
    i32 i_modes_d[4] = {0};
    i32 i_modes_e[4] = {0};

    for (i32 i = 0; i < a.num_modes; i++) i_modes_a[i] = (i32)modes_a[i];
    for (i32 i = 0; i < b.num_modes; i++) i_modes_b[i] = (i32)modes_b[i];
    for (i32 i = 0; i < c.num_modes; i++) i_modes_c[i] = (i32)modes_c[i];
    for (i32 i = 0; i < d.num_modes; i++) i_modes_d[i] = (i32)modes_d[i];
    for (i32 i = 0; i < e.num_modes; i++) i_modes_e[i] = (i32)modes_e[i];

    cutensorTensorDescriptor_t cutensor_desc_a = cutensor_tensor_descriptor_create(cutensor, a);
    cutensorTensorDescriptor_t cutensor_desc_b = cutensor_tensor_descriptor_create(cutensor, b);
    cutensorTensorDescriptor_t cutensor_desc_c = cutensor_tensor_descriptor_create(cutensor, c);
    cutensorTensorDescriptor_t cutensor_desc_d = cutensor_tensor_descriptor_create(cutensor, d);
    cutensorTensorDescriptor_t cutensor_desc_e = cutensor_tensor_descriptor_create(cutensor, e);

    cutensorOperationDescriptor_t cutensor_desc;
    cutensorCheck(
        cutensorCreateContractionTrinary(
            cutensor.handle,
            &cutensor_desc,
            cutensor_desc_a, i_modes_a, CUTENSOR_OP_IDENTITY,
            cutensor_desc_b, i_modes_b, CUTENSOR_OP_IDENTITY,
            cutensor_desc_c, i_modes_c, CUTENSOR_OP_IDENTITY,
            cutensor_desc_d, i_modes_d, CUTENSOR_OP_IDENTITY,
            cutensor_desc_e, i_modes_e,
            cutensor_desc_compute
        )
    );

    cutensorDataType_t scalar_type;
    cutensorCheck(
        cutensorOperationDescriptorGetAttribute(
            cutensor.handle,
            cutensor_desc,
            CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
            (void*)&scalar_type,
            sizeof(scalar_type)
        )
    );

    if constexpr (std::is_same_v<T, f32>) {
        ASSERT( scalar_type == CUTENSOR_R_32F );
    } else if constexpr (std::is_same_v<T, f64>) {
        ASSERT( scalar_type == CUTENSOR_R_64F ); 
    }

    const cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t cutensor_plan_pref;
    cutensorCheck(
        cutensorCreatePlanPreference(
            cutensor.handle,
            &cutensor_plan_pref,
            cutensor_algo,
            JIT_MODE
        )
    );

    u64 cutensor_workspace_size_estimate = 0;
    const cutensorWorksizePreference_t cutensor_workspace_pref = CUTENSOR_WORKSPACE_MAX;
    cutensorCheck(
        cutensorEstimateWorkspaceSize(
            cutensor.handle,
            cutensor_desc,
            cutensor_plan_pref,
            cutensor_workspace_pref,
            &cutensor_workspace_size_estimate
        )
    );

    cutensorPlan_t cutensor_plan;
    cutensorCheck(
        cutensorCreatePlan(
            cutensor.handle,
            &cutensor_plan,
            cutensor_desc,
            cutensor_plan_pref,
            cutensor_workspace_size_estimate
        )
    );

    u64 cutensor_actual_workspace_size = 0;
    cutensorCheck(
        cutensorPlanGetAttribute(
            cutensor.handle,
            cutensor_plan,
            CUTENSOR_PLAN_REQUIRED_WORKSPACE,
            &cutensor_actual_workspace_size,
            sizeof(cutensor_actual_workspace_size)
        )
    );

    DeviceMemory cutensor_workspace(dsa, cutensor_actual_workspace_size, DEVICE_ALLOC_ALIGNMENT);

    if (flops_ref != NULL) {
        f32 flops;
        cutensorCheck(
            cutensorOperationDescriptorGetAttribute(
                cutensor.handle,
                cutensor_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
                (void*)&flops,
                sizeof(flops)
            )
        );
        *flops_ref = flops;
    }

    WET_COND(dsa) {
        cutensorCheck(
            cutensorContractTrinary(
                cutensor.handle,
                cutensor_plan,
                (void*) &alpha, a.tensor.ptr(), b.tensor.ptr(), c.tensor.ptr(),
                (void*) &beta, d.tensor.ptr(), e.tensor.ptr(),
                cutensor_workspace.ptr(),
                cutensor_actual_workspace_size,
                stream
            )
        );
    }

    cutensorCheck( cutensorDestroyPlan(cutensor_plan) );
    cutensorCheck( cutensorDestroyPlanPreference(cutensor_plan_pref) );
    cutensorCheck( cutensorDestroyOperationDescriptor(cutensor_desc) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_e) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_d) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_c) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_b) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_a) );
}

template <class T>
static
void
cutensor_elementwise_binary(
    CUTensor &cutensor,
    DeviceStackAllocator &dsa,  
    T alpha, 
    DeviceMultiTensor<T> &a, 
    const char *modes_a, 
    cutensorOperator_t op_a,
    T gamma,  
    DeviceMultiTensor<T> &c, 
    const char *modes_c,
    cutensorOperator_t op_c,
    DeviceMultiTensor<T> &d,
    cutensorOperator_t op_ac,
    cudaStream_t stream,
    f32 *flops_ref = NULL
) {
    if (strlen(modes_a) == 0) {
        ASSERT( a.num_modes == 1 && a.extent[0] == 1);
    } else {
        ASSERT( strlen(modes_a) == a.num_modes );
    }

    if(strlen(modes_c) == 0) {
        ASSERT( c.num_modes == 1 && c.extent[0] == 1);
    } else {
        ASSERT( strlen(modes_c) == c.num_modes );
    }

    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>, "Unsupported floating type");

    cutensorComputeDescriptor_t cutensor_desc_compute;
    if constexpr (std::is_same_v<T, f32>) {
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_32F;
    } else if constexpr (std::is_same_v<T, f64>) {
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_64F;
    }

    i32 i_modes_a[4] = {0};
    i32 i_modes_c[4] = {0};

    for (i32 i = 0; i < a.num_modes; i++) i_modes_a[i] = (i32)modes_a[i];
    for (i32 i = 0; i < c.num_modes; i++) i_modes_c[i] = (i32)modes_c[i];

    cutensorTensorDescriptor_t cutensor_desc_a = cutensor_tensor_descriptor_create(cutensor, a);
    cutensorTensorDescriptor_t cutensor_desc_c = cutensor_tensor_descriptor_create(cutensor, c);
    cutensorTensorDescriptor_t cutensor_desc_d = cutensor_tensor_descriptor_create(cutensor, d);

    cutensorOperationDescriptor_t cutensor_desc;
    cutensorCheck( 
        cutensorCreateElementwiseBinary(
            cutensor.handle,
            &cutensor_desc,
            cutensor_desc_a, i_modes_a, op_a,
            cutensor_desc_c, i_modes_c, op_c,
            cutensor_desc_d, i_modes_c,
            op_ac,
            cutensor_desc_compute
        )
    );

    cutensorDataType_t scalar_type;
    cutensorCheck(
        cutensorOperationDescriptorGetAttribute(
            cutensor.handle,
            cutensor_desc,
            CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
            (void*)&scalar_type,
            sizeof(scalar_type)
        )
    );

    if constexpr (std::is_same_v<T, f32>) {
        ASSERT( scalar_type == CUTENSOR_R_32F );
    } else if constexpr (std::is_same_v<T, f64>) {
        ASSERT( scalar_type == CUTENSOR_R_64F ); 
    }

    const cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t cutensor_plan_pref;
    cutensorCheck(
        cutensorCreatePlanPreference(
            cutensor.handle,
            &cutensor_plan_pref,
            cutensor_algo,
            JIT_MODE
        )
    );

    cutensorPlan_t cutensor_plan;
    cutensorCheck(
        cutensorCreatePlan(
            cutensor.handle,
            &cutensor_plan,
            cutensor_desc,
            cutensor_plan_pref,
            0
        )
    );

    if (flops_ref != NULL) {
        f32 flops;
        cutensorCheck(
            cutensorOperationDescriptorGetAttribute(
                cutensor.handle,
                cutensor_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
                (void*)&flops,
                sizeof(flops)
            )
        );
        *flops_ref = flops;
    }

    WET_COND(dsa) {
        cutensorCheck(
            cutensorElementwiseBinaryExecute(
                cutensor.handle,
                cutensor_plan,
                (void*) &alpha, a.tensor.ptr(), 
                (void*) &gamma, c.tensor.ptr(),
                d.tensor.ptr(),
                stream
            )
        );
    }

    cutensorCheck( cutensorDestroyPlan(cutensor_plan) );
    cutensorCheck( cutensorDestroyPlanPreference(cutensor_plan_pref) );
    cutensorCheck( cutensorDestroyOperationDescriptor(cutensor_desc) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_d) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_c) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_a) );
}

// Seems like only the modes for c and d need to be an identical pointer
template <class T>
static
void
cutensor_elementwise_trinary(
    CUTensor &cutensor,
    DeviceStackAllocator &dsa,  
    T alpha, 
    DeviceMultiTensor<T> &a, 
    const char *modes_a, 
    cutensorOperator_t op_a,
    T beta,  
    DeviceMultiTensor<T> &b, 
    const char *modes_b,
    cutensorOperator_t op_b,
    T gamma, 
    DeviceMultiTensor<T> &c, 
    const char *modes_c,
    cutensorOperator_t op_c,
    DeviceMultiTensor<T> &d,
    cutensorOperator_t op_ab,
    cutensorOperator_t op_abc,
    cudaStream_t stream,
    f32 *flops_ref = NULL
) {
    ASSERT( strlen(modes_a) == a.num_modes );
    ASSERT( strlen(modes_b) == b.num_modes );
    ASSERT( strlen(modes_c) == c.num_modes );

    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>, "Unsupported floating type");

    cutensorComputeDescriptor_t cutensor_desc_compute;
    if constexpr (std::is_same_v<T, f32>) {
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_32F;
    } else if constexpr (std::is_same_v<T, f64>) {
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_64F;
    }

    i32 i_modes_a[4] = {0};
    i32 i_modes_b[4] = {0};
    i32 i_modes_c[4] = {0};

    for (i32 i = 0; i < a.num_modes; i++) i_modes_a[i] = (i32)modes_a[i];
    for (i32 i = 0; i < b.num_modes; i++) i_modes_b[i] = (i32)modes_b[i];
    for (i32 i = 0; i < c.num_modes; i++) i_modes_c[i] = (i32)modes_c[i];

    cutensorTensorDescriptor_t cutensor_desc_a = cutensor_tensor_descriptor_create(cutensor, a);
    cutensorTensorDescriptor_t cutensor_desc_b = cutensor_tensor_descriptor_create(cutensor, b);
    cutensorTensorDescriptor_t cutensor_desc_c = cutensor_tensor_descriptor_create(cutensor, c);
    cutensorTensorDescriptor_t cutensor_desc_d = cutensor_tensor_descriptor_create(cutensor, d);

    cutensorOperationDescriptor_t cutensor_desc;
    cutensorCheck( 
        cutensorCreateElementwiseTrinary(
            cutensor.handle,
            &cutensor_desc,
            cutensor_desc_a, i_modes_a, op_a,
            cutensor_desc_b, i_modes_b, op_b,
            cutensor_desc_c, i_modes_c, op_c,
            cutensor_desc_d, i_modes_c,
            op_ab,
            op_abc,
            cutensor_desc_compute
        )
    );

    cutensorDataType_t scalar_type;
    cutensorCheck(
        cutensorOperationDescriptorGetAttribute(
            cutensor.handle,
            cutensor_desc,
            CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
            (void*)&scalar_type,
            sizeof(scalar_type)
        )
    );

    if constexpr (std::is_same_v<T, f32>) {
        ASSERT( scalar_type == CUTENSOR_R_32F );
    } else if constexpr (std::is_same_v<T, f64>) {
        ASSERT( scalar_type == CUTENSOR_R_64F ); 
    }

    const cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t cutensor_plan_pref;
    cutensorCheck(
        cutensorCreatePlanPreference(
            cutensor.handle,
            &cutensor_plan_pref,
            cutensor_algo,
            JIT_MODE
        )
    );

    cutensorPlan_t cutensor_plan;
    cutensorCheck(
        cutensorCreatePlan(
            cutensor.handle,
            &cutensor_plan,
            cutensor_desc,
            cutensor_plan_pref,
            0
        )
    );

    if (flops_ref != NULL) {
        f32 flops;
        cutensorCheck(
            cutensorOperationDescriptorGetAttribute(
                cutensor.handle,
                cutensor_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
                (void*)&flops,
                sizeof(flops)
            )
        );
        *flops_ref = flops;
    }

    WET_COND(dsa) {
        cutensorCheck(
            cutensorElementwiseTrinaryExecute(
                cutensor.handle,
                cutensor_plan,
                (void*) &alpha, a.tensor.ptr(), 
                (void*) &beta, b.tensor.ptr(),
                (void*) &gamma, c.tensor.ptr(),
                d.tensor.ptr(),
                stream
            )
        );
    }

    cutensorCheck( cutensorDestroyPlan(cutensor_plan) );
    cutensorCheck( cutensorDestroyPlanPreference(cutensor_plan_pref) );
    cutensorCheck( cutensorDestroyOperationDescriptor(cutensor_desc) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_c) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_b) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_a) );
}

template <class T>
static
void
cutensor_reduction(
    CUTensor &cutensor,
    DeviceStackAllocator &dsa,
    T alpha,
    DeviceMultiTensor<T> &a,
    const char *modes_a,
    cutensorOperator_t op_a,
    T beta,
    DeviceMultiTensor<T> &c,
    const char *modes_c,
    cutensorOperator_t op_c,
    DeviceMultiTensor<T> &d,
    cutensorOperator_t op_reduce,
    cudaStream_t stream,
    f32 *flops_ref = NULL
) {
    ASSERT( uintptr_t(a.ptr()) % DEVICE_ALLOC_ALIGNMENT == 0 );
    ASSERT( uintptr_t(c.ptr()) % DEVICE_ALLOC_ALIGNMENT == 0 );
    ASSERT( uintptr_t(d.ptr()) % DEVICE_ALLOC_ALIGNMENT == 0 );

    ASSERT( strlen(modes_a) == a.num_modes );

    if(strlen(modes_c) == 0) {
        ASSERT( c.num_modes == 1 && c.extent[0] == 1);
    } else {
        ASSERT( strlen(modes_c) == c.num_modes );
    }

    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>, "Unsupported floating type");

    cutensorComputeDescriptor_t cutensor_desc_compute;
    if constexpr (std::is_same_v<T, f32>) {
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_32F;
    } else if constexpr (std::is_same_v<T, f64>) {
        cutensor_desc_compute = CUTENSOR_COMPUTE_DESC_64F;
    }

    i32 i_modes_a[4] = {0};
    i32 i_modes_c[4] = {0};

    for (i32 i = 0; i < a.num_modes; i++) i_modes_a[i] = (i32)modes_a[i];
    for (i32 i = 0; i < c.num_modes; i++) i_modes_c[i] = (i32)modes_c[i];

    cutensorTensorDescriptor_t cutensor_desc_a = cutensor_tensor_descriptor_create(cutensor, a);
    cutensorTensorDescriptor_t cutensor_desc_c = cutensor_tensor_descriptor_create(cutensor, c);
    cutensorTensorDescriptor_t cutensor_desc_d = cutensor_tensor_descriptor_create(cutensor, d);

    cutensorOperationDescriptor_t cutensor_desc;
    cutensorCheck( 
        cutensorCreateReduction(
            cutensor.handle,
            &cutensor_desc,
            cutensor_desc_a, i_modes_a, op_a,
            cutensor_desc_c, i_modes_c, op_c,
            cutensor_desc_d, i_modes_c,
            op_reduce,
            cutensor_desc_compute
        )
    );

    cutensorDataType_t scalar_type;
    cutensorCheck(
        cutensorOperationDescriptorGetAttribute(
            cutensor.handle,
            cutensor_desc,
            CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
            (void*)&scalar_type,
            sizeof(scalar_type)
        )
    );

    if constexpr (std::is_same_v<T, f32>) {
        ASSERT( scalar_type == CUTENSOR_R_32F );
    } else if constexpr (std::is_same_v<T, f64>) {
        ASSERT( scalar_type == CUTENSOR_R_64F ); 
    }

    const cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t cutensor_plan_pref;
    cutensorCheck(
        cutensorCreatePlanPreference(
            cutensor.handle,
            &cutensor_plan_pref,
            cutensor_algo,
            JIT_MODE
        )
    );

    u64 cutensor_workspace_size_estimate = 0;
    const cutensorWorksizePreference_t cutensor_workspace_pref = CUTENSOR_WORKSPACE_MAX;
    cutensorCheck(
        cutensorEstimateWorkspaceSize(
            cutensor.handle,
            cutensor_desc,
            cutensor_plan_pref,
            cutensor_workspace_pref,
            &cutensor_workspace_size_estimate
        )
    );

    cutensorPlan_t cutensor_plan;
    cutensorCheck(
        cutensorCreatePlan(
            cutensor.handle,
            &cutensor_plan,
            cutensor_desc,
            cutensor_plan_pref,
            0
        )
    );

    u64 cutensor_actual_workspace_size = 0;
    cutensorCheck(
        cutensorPlanGetAttribute(
            cutensor.handle,
            cutensor_plan,
            CUTENSOR_PLAN_REQUIRED_WORKSPACE,
            &cutensor_actual_workspace_size,
            sizeof(cutensor_actual_workspace_size)
        )
    );

    DeviceMemory cutensor_workspace(dsa, cutensor_actual_workspace_size, DEVICE_ALLOC_ALIGNMENT);

    ASSERT( uintptr_t(cutensor_workspace.ptr()) % DEVICE_ALLOC_ALIGNMENT == 0 );

    if (flops_ref != NULL) {
        f32 flops;
        cutensorCheck(
            cutensorOperationDescriptorGetAttribute(
                cutensor.handle,
                cutensor_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
                (void*)&flops,
                sizeof(flops)
            )
        );
        *flops_ref = flops;
    }

    WET_COND(dsa) {
        cutensorCheck(
            cutensorReduce(
                cutensor.handle,
                cutensor_plan,
                (void*) &alpha, a.tensor.ptr(), 
                (void*) &gamma, c.tensor.ptr(),
                d.tensor.ptr(),
                cutensor_workspace.ptr(),
                cutensor_actual_workspace_size,
                stream
            )
        );
    }

    cutensorCheck( cutensorDestroyPlan(cutensor_plan) );
    cutensorCheck( cutensorDestroyPlanPreference(cutensor_plan_pref) );
    cutensorCheck( cutensorDestroyOperationDescriptor(cutensor_desc) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_d) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_c) );
    cutensorCheck( cutensorDestroyTensorDescriptor(cutensor_desc_a) );
}

}
