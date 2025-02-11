#include "compat.cuh"
#include "../config.h"
#include "matrix_view.cuh"
#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"

#define EXL2_BLOCK_M_SIZE_MAX MAX_Q_GEMM_ROWS_KERNEL

//#define EXL2_BLOCK_KN_SIZE 32
//#define EXL2_MAX_GROUPS_IN_BLOCK (EXL2_BLOCK_KN_SIZE / 32)

__forceinline__ __device__ half2 dot22_8(half2(&dq)[4], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_16(half2(&dq)[8], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_32(half2(&dq)[16], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ float dot22_8_f(half2(&dq)[4], const half* a_ptr, const float g_result, const float qs_f)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    float result_f = __half2float(__low2half(result)) + __half2float(__high2half(result));
    return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_16_f(half2(&dq)[8], const half* a_ptr, const float g_result, const float qs_f)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    float result_f = __half2float(__low2half(result)) + __half2float(__high2half(result));
    return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_32_f(half2(&dq)[16], const half* a_ptr, const float g_result, const float qs_f)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
    float result_f = __half2float(__low2half(result)) + __half2float(__high2half(result));
    return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ half dot22_8_h(half2(&dq)[4], const half* a_ptr, const half g_result, const half qs_h)
{
    // Use FP32 accumulator to avoid potential overflow since unscaled weights are in the range -128..127

    float result = {};
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        half2 w01 = dq[i];
        float w0 = __low2float(w01);
        float w1 = __high2float(w01);
        float x0 = __half2float(*a_ptr++);
        float x1 = __half2float(*a_ptr++);
        result = fma(w0, x0, result);
        result = fma(w1, x1, result);
    }
    float qs = __half2float(qs_h);
    result *= qs;
    half result_h = __float2half_rn(result);
    return __hadd(result_h, g_result);
}

__forceinline__ __device__ half dot22_16_h(half2(&dq)[8], const half* a_ptr, const half g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    half result_h = __hadd(__low2half(result), __high2half(result));
    return __hfma(result_h, qs_h, g_result);
}

__forceinline__ __device__ half dot22_32_h(half2(&dq)[16], const half* a_ptr, const half g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
    half result_h = __hadd(__low2half(result), __high2half(result));
    return __hfma(result_h, qs_h, g_result);
}

__forceinline__ __device__ half dot22_32_pc(half2(&dq)[16], const half* a_ptr, const half g_result)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
    half result_h = __hadd(__low2half(result), __high2half(result));
    return __hadd(result_h, g_result);
}


typedef void (*fp_gemm_half_q_half_kernel)
(
    const half*,
    const uint32_t*,
    const uint32_t*,
    const half*,
    half*,
    const int,
    const int,
    const int,
    const int,
    const uint16_t*,
    const uint16_t*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const bool,
    const half*,
    const int
);
__device__ static float nf5_data[32] = {-1.0000, -0.8258, -0.7103, -0.6203, -0.5448, -0.4786, -0.4190, -0.3640,-0.3126, -0.2638, -0.2171, -0.1720, -0.1281, -0.0849, -0.0423,  0.0000,0.0397,  0.0796,  0.1199,  0.1609,  0.2029,  0.2461,  0.2910,  0.3379,0.3876,  0.4407,  0.4985,  0.5626,  0.6358,  0.7230,  0.8344,  1.0000};

template <int m_count, int kernel_p, bool use_r_weights, bool mul_r_weights, int block_kn_size>
__global__ void gemm_half_q_half_kernel
(
    const half*      __restrict__ a,
    const uint32_t*  __restrict__ b_q_weight,
    const uint32_t*  __restrict__ b_q_scale,
    const half*      __restrict__ b_q_scale_max,
    half*            __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const uint16_t* __restrict__ b_q_group_map,
    const uint16_t* __restrict__ b_q_perm,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2,
    const bool clear,
    const half* r_weights,
    const int r_weights_stride
)
{
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);

    int t = threadIdx.x;
    __shared__ half quant_data[32];
    for (int i = 0; i < 32; i++) {
        quant_data[i] = __float2half(nf5_data[i]);
    }

    // Block

//    const int m_count = EXL2_BLOCK_M_SIZE_MAX;

    int offset_n = blockIdx.x * block_kn_size * 4;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * block_kn_size;

    int m_count_min = min(size_m - offset_m, m_count);

    int end_n = min(offset_n + block_kn_size * 4, size_n);
    int end_m = min(offset_m + m_count_min, size_m);
    int end_k = min(offset_k + block_kn_size, size_k);
    int n = offset_n + t * 4;

    // Read weights

    half_uint16 weights[MAX_Q_GEMM_WEIGHTS];
    if constexpr (use_r_weights)
    {
        uint16_t any_w = 0;
        const half* w_ptr = r_weights;
        for (int m = 0; m < m_count_min; ++m)
        {
            weights[m].as_half = *w_ptr;
            w_ptr += r_weights_stride;
            any_w |= weights[m].as_uint16;
        }
        if (!any_w) return;  // Early exit if all weights are zero -- does not zero output (!!!)
    }

    // Preload block_a

    __shared__ half block_a[m_count][block_kn_size];

    if (offset_k + t < end_k)
    {
        for (int m = 0; m < m_count_min; ++m)
        {
            const half* a_ptr = a_.item_ptr(offset_m + m, 0);
            half* block_a_ptr = block_a[m];
            half a0;
            if (b_q_perm)
                a0 = a_ptr[b_q_perm[offset_k + t]];
            else
                a0 = a_ptr[offset_k + t];
            block_a_ptr[t] = a0;
        }
    }

    // Clear

    if (n >= size_n) return;

    if (clear && blockIdx.z == 0) // && (threadIdx.x & 1) == 0)
    {
        for (int m = 0; m < m_count_min; m++)
            *((uint64_t*) c_.item_ptr(offset_m + m, n)) = 0;
    }

    __syncthreads();

    // Find initial group

    //int group = offset_k / groupsize;
    int group = b_q_group_map[offset_k * 2];

//    if (offset_m == 0 && t == 0)
//        DBGI2(offset_k, group);

    // a, b offset

    int pre_rows_8 = min(rows_8, offset_k);
    int pre_rows_6 = offset_k > rows_8 ? min(rows_6, offset_k) - rows_8 : 0;
    int pre_rows_5 = offset_k > rows_6 ? min(rows_5, offset_k) - rows_6 : 0;
    int pre_rows_4 = offset_k > rows_5 ? min(rows_4, offset_k) - rows_5 : 0;
    int pre_rows_3 = offset_k > rows_4 ? min(rows_3, offset_k) - rows_4 : 0;
    int pre_rows_2 = offset_k > rows_3 ? min(rows_2, offset_k) - rows_3 : 0;
    int qk = 0;
    qk += pre_rows_8 / 32 * 8;
    qk += pre_rows_6 / 32 * 6;
    qk += pre_rows_5 / 32 * 5;
    qk += pre_rows_4 / 32 * 4;
    qk += pre_rows_3 / 32 * 3;
    qk += pre_rows_2 / 32 * 2;

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
    const half* a_ptr = &block_a[0][0];
    int a_stride = block_kn_size;

    // Column result

    half block_c[m_count][4] = {};

    // Dequantize groups

    int k = offset_k;
    if constexpr (kernel_p & 0b00010000) {
    while (k < rows_5 && k < end_k)
    {
        #pragma unroll
        for (int j = 0; j < 1; j++)
        {
            int4 load_int4[5];
            load_int4[0] = *((int4*) b_ptr); b_ptr += size_n;
            load_int4[1] = *((int4*) b_ptr); b_ptr += size_n;
            load_int4[2] = *((int4*) b_ptr); b_ptr += size_n;
            load_int4[3] = *((int4*) b_ptr); b_ptr += size_n;
            load_int4[4] = *((int4*) b_ptr); b_ptr += size_n;

            half2 dq[4][16];
            dequant_5bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, load_int4[3].x, load_int4[4].x, dq[0], size_n, quant_data);
            dequant_5bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, load_int4[3].y, load_int4[4].y, dq[1], size_n, quant_data);
            dequant_5bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, load_int4[3].z, load_int4[4].z, dq[2], size_n, quant_data);
            dequant_5bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, load_int4[3].w, load_int4[4].w, dq[3], size_n, quant_data);

            for (int m = 0; m < m_count_min; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }
                block_c[m][0] = dot22_32_pc(dq[0], a_ptr + m * a_stride, block_c[m][0]);
                block_c[m][1] = dot22_32_pc(dq[1], a_ptr + m * a_stride, block_c[m][1]);
                block_c[m][2] = dot22_32_pc(dq[2], a_ptr + m * a_stride, block_c[m][2]);
                block_c[m][3] = dot22_32_pc(dq[3], a_ptr + m * a_stride, block_c[m][3]);
            }
            a_ptr += 32;
        }

        k += 32;
    }}


    // Accumulate column sums in c

    for (int m = 0; m < m_count_min; m++)
    {
        half2* out = (half2*)c_.item_ptr(offset_m + m, n);
        half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
        half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);

        if constexpr (mul_r_weights)
        {
            half2 w_mul2 = __half2half2(weights[m].as_half);
            result01 = __hmul2(result01, w_mul2);
            result23 = __hmul2(result23, w_mul2);
        }

        atomicAdd(out    , result01);
        atomicAdd(out + 1, result23);
    }
}

