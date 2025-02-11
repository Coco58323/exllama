import torch
import torch.utils.cpp_extension

cuda_src = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// Constants for kernel configuration
#define BLOCK_KN_SIZE 32
#define MAX_Q_GEMM_ROWS 32
#define THREADS_X 32
#define THREADS_Y 32

// Helper struct for half2 and uint32 union
union half_uint16 {
    half as_half;
    uint16_t as_uint16;
};

// Helper struct for matrix view
template<typename T>
struct MatrixView {
    T* data;
    int rows;
    int cols;
    
    __device__ T* item_ptr(int row, int col) {
        return data + row * cols + col;
    }
};

// Dequantization functions for different bit widths
__device__ void dequant_4bit_8(uint32_t q, half2 (&dq)[4], int stride) {
    // Unpack 4-bit values and convert to FP16
    uint32_t packed = q;
    for (int i = 0; i < 4; i++) {
        uint32_t vals = packed & 0x0F0F0F0F;
        packed >>= 4;
        half2 fp_vals;
        fp_vals.x = __int2half_rn(vals & 0xFF);
        fp_vals.y = __int2half_rn((vals >> 16) & 0xFF); 
        dq[i] = fp_vals;
    }
}

// Main quantized matrix multiplication kernel
template<int m_count>
__global__ void gemm_half_q_half_kernel(
    const half* __restrict__ a,           // Input matrix A
    const uint32_t* __restrict__ b_q,     // Quantized matrix B 
    const uint32_t* __restrict__ b_scale, // Scales for B
    const half* __restrict__ b_scale_max, // Max scales for B
    half* __restrict__ c,                 // Output matrix C
    const int size_m,                     // M dimension
    const int size_n,                     // N dimension  
    const int size_k,                     // K dimension
    const int groups,                     // Number of groups
    const uint16_t* __restrict__ group_map, // Group mapping
    const bool clear                      // Clear output flag
) {
    // Create matrix views
    MatrixView<const half> a_view{a, size_m, size_k};
    MatrixView<half> c_view{c, size_m, size_n}; 

    // Calculate offsets
    int t = threadIdx.x;
    int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * BLOCK_KN_SIZE;

    // Bounds checking
    int m_count_min = min(size_m - offset_m, m_count);
    if (offset_n >= size_n) return;

    // Shared memory for block of A
    __shared__ half block_a[m_count][BLOCK_KN_SIZE];

    // Load block of A into shared memory
    if (offset_k + t < size_k) {
        for (int m = 0; m < m_count_min; m++) {
            block_a[m][t] = a_view.item_ptr(offset_m + m, offset_k + t)[0];
        }
    }
    __syncthreads();

    // Clear output if needed
    if (clear && blockIdx.z == 0) {
        for (int m = 0; m < m_count_min; m++) {
            *((uint64_t*)c_view.item_ptr(offset_m + m, offset_n + t * 4)) = 0;
        }
    }

    // Main computation loop
    int k = offset_k;
    const half* a_ptr = &block_a[0][0];
    const uint32_t* b_ptr = b_q + (k/8) * size_n + offset_n + t * 4;
    
    while (k < size_k) {
        // Dequantize block of B
        half2 dq[4][4];
        uint32_t q = *b_ptr;
        dequant_4bit_8(q, dq[0], size_n);
        
        // Matrix multiply for each row in the block
        for (int m = 0; m < m_count_min; m++) {
            half2 result = {};
            const half2* a2_ptr = (const half2*)(a_ptr + m * BLOCK_KN_SIZE);
            
            // Dot product
            for (int i = 0; i < 4; i++) {
                result = __hfma2(dq[0][i], *a2_ptr++, result);
            }
            
            // Scale and accumulate result
            half scale = b_scale_max[k/BLOCK_KN_SIZE];
            half* c_out = c_view.item_ptr(offset_m + m, offset_n + t * 4);
            atomicAdd(c_out, __hmul(__low2half(result), scale));
            atomicAdd(c_out + 1, __hmul(__high2half(result), scale));
        }
        
        k += 8;
        b_ptr += size_n;
        a_ptr += 8;
    }
}

// C++ wrapper function
torch::Tensor exllama_gemm(
    const torch::Tensor& a,      // Input matrix A (FP16)
    const torch::Tensor& b_q,    // Quantized matrix B 
    const torch::Tensor& b_scale // Scales for B
) {
    // Input validation
    TORCH_CHECK(a.is_cuda(), "Matrix A must be a CUDA tensor");
    TORCH_CHECK(b_q.is_cuda(), "Matrix B must be a CUDA tensor");
    TORCH_CHECK(b_scale.is_cuda(), "Scales must be a CUDA tensor");
    
    // Get dimensions
    int m = a.size(0);
    int k = a.size(1);
    int n = b_q.size(1) * 8; // Each uint32 contains 8 4-bit values
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(a.device());
    auto c = torch::empty({m, n}, options);
    
    // Configure kernel launch
    dim3 threads(THREADS_X, 1);
    dim3 blocks(
        (n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
        (m + MAX_Q_GEMM_ROWS - 1) / MAX_Q_GEMM_ROWS,
        (k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE
    );
    
    // Launch kernel
    gemm_half_q_half_kernel<MAX_Q_GEMM_ROWS><<<blocks, threads>>>(
        a.data_ptr<half>(),
        b_q.data_ptr<uint32_t>(),
        b_scale.data_ptr<uint32_t>(),
        nullptr, // b_scale_max not used in this simplified version
        c.data_ptr<half>(),
        m, n, k,
        1,  // groups
        nullptr, // group_map not used
        true // clear output
    );
    
    return c;
}
"""

cpp_src = """
torch::Tensor exllama_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b_q, 
    const torch::Tensor& b_scale
);
"""

exllama_gemm_module = torch.utils.cpp_extension.load_inline(
    name="exllama_gemm",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=['exllama_gemm'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

def quantized_matmul(a: torch.Tensor, b_q: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication with a quantized weight matrix.
    
    Args:
        a: Input tensor in FP16 format (M x K)
        b_q: Quantized weight matrix (K/8 x N) in uint32 format
        b_scale: Scale factors for quantized values
        
    Returns:
        Result tensor in FP16 format (M x N)
    """
    return exllama_gemm_module.exllama_gemm(a, b_q, b_scale) 