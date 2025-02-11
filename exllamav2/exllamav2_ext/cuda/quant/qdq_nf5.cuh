#ifndef _qdq_5_cuh
#define _qdq_5_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_5BIT == 1
// __device__ static float nf4_data[16] = {-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0};
// Permutation:
//
// v5555533 33311111  u4444422 22200000  (u, v lsb)
// vbbbbb99 99977777  uaaaaa88 88866666
// vhhhhhff fffddddd  ugggggee eeeccccc
// vnnnnnll llljjjjj  ummmmmkk kkkiiiii
// vtttttrr rrrppppp  usssssqq qqqooooo

__forceinline__ __device__ void shuffle_5bit_32
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0 * stride];
    uint32_t qb = q[1 * stride];
    uint32_t qc = q[2 * stride];
    uint32_t qd = q[3 * stride];
    uint32_t qe = q[4 * stride];

    // qa: 66555554 44443333  32222211 11100000
    // qb: ccccbbbb baaaaa99  99988888 77777666
    // qc: jiiiiihh hhhggggg  fffffeee eedddddc
    // qd: pppooooo nnnnnmmm  mmlllllk kkkkjjjj
    // qe: vvvvvuuu uuttttts  ssssrrrr rqqqqqpp

    uint32_t qf = qe >> 22;
    qe <<= 8;
    qe |= qd >> 24;
    qd <<= 6;
    qd |= qc >> 26;
    qc <<= 4;
    qc |= qb >> 28;
    qb <<= 2;
    qb |= qa >> 30;

    // qa:   555554 44443333  32222211 11100000
    // qb:   bbbbba aaaa9999  98888877 77766666
    // qc:   hhhhhg ggggffff  feeeeedd dddccccc
    // qd:   nnnnnm mmmmllll  lkkkkkjj jjjiiiii
    // qe:   ttttts ssssrrrr  rqqqqqpp pppooooo
    // qf:                          vv vvvuuuuu

    uint32_t za = 0;
    uint32_t zb = 0;
    uint32_t zc = 0;
    uint32_t zd = 0;
    uint32_t ze = 0;

    for (int i = 0; i < 3; i++) { uint32_t t0 = qa & 0x1f; uint32_t t1 = (qa & 0x3e0) >> 5; qa >>= 10; za |= (t0 << (i * 5)); za |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qb & 0x1f; uint32_t t1 = (qb & 0x3e0) >> 5; qb >>= 10; zb |= (t0 << (i * 5)); zb |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qc & 0x1f; uint32_t t1 = (qc & 0x3e0) >> 5; qc >>= 10; zc |= (t0 << (i * 5)); zc |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qd & 0x1f; uint32_t t1 = (qd & 0x3e0) >> 5; qd >>= 10; zd |= (t0 << (i * 5)); zd |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qe & 0x1f; uint32_t t1 = (qe & 0x3e0) >> 5; qe >>= 10; ze |= (t0 << (i * 5)); ze |= (t1 << (i * 5 + 16)); }

    // za:  5555533 33311111   4444422 22200000
    // zb:  bbbbb99 99977777   aaaaa88 88866666
    // zc:  hhhhhff fffddddd   gggggee eeeccccc
    // zd:  nnnnnll llljjjjj   mmmmmkk kkkiiiii
    // ze:  tttttrr rrrppppp   sssssqq qqqooooo
    // qf:                          vv vvvuuuuu

    za |= ((qf & 0x001) >> 0) << 15;
    zb |= ((qf & 0x002) >> 1) << 15;
    zc |= ((qf & 0x004) >> 2) << 15;
    zd |= ((qf & 0x008) >> 3) << 15;
    ze |= ((qf & 0x010) >> 4) << 15;
    za |= ((qf & 0x020) >> 5) << 31;
    zb |= ((qf & 0x040) >> 6) << 31;
    zc |= ((qf & 0x080) >> 7) << 31;
    zd |= ((qf & 0x100) >> 8) << 31;
    ze |= ((qf & 0x200) >> 9) << 31;

    // za: v5555533 33311111  u4444422 22200000  (u, v lsb)
    // zb: vbbbbb99 99977777  uaaaaa88 88866666
    // zc: vhhhhhff fffddddd  ugggggee eeeccccc
    // zd: vnnnnnll llljjjjj  ummmmmkk kkkiiiii
    // ze: vtttttrr rrrppppp  usssssqq qqqooooo

    q[0 * stride] = za;
    q[1 * stride] = zb;
    q[2 * stride] = zc;
    q[3 * stride] = zd;
    q[4 * stride] = ze;
}

__forceinline__ __device__ void dequant_5bit_32
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    const uint32_t q_3,
    const uint32_t q_4,
    half2 (&dq)[16],
    int stride,
    half* nf5_data
)
{
    // v5555533 33311111  u4444422 22200000  (u, v lsb)
    // vbbbbb99 99977777  uaaaaa88 88866666
    // vhhhhhff fffddddd  ugggggee eeeccccc
    // vnnnnnll llljjjjj  ummmmmkk kkkiiiii
    // vtttttrr rrrppppp  usssssqq qqqooooo
    // for (int i = 0; i < 16; i++) {
    //     half dq1 = nf5_data[q_0>>(i*5) & 0x1f];
    //     half dq2 = nf5_data[q_1>>(i*5) & 0x1f];
    //     dq[i] = __halves2half2(dq1, dq2);
    // }
    half dq0 = nf5_data[q_0 & 0x1f];
    half dq1 = nf5_data[q_0>>16 & 0x1f];
    half dq2 = nf5_data[q_0>>5 & 0x1f];
    half dq3 = nf5_data[q_0>>21 & 0x1f];
    half dq4 = nf5_data[q_0>>10 & 0x1f];
    half dq5 = nf5_data[q_0>>26 & 0x1f];
    half dq6 = nf5_data[q_1 & 0x1f];
    half dq7 = nf5_data[q_1>>16 & 0x1f];
    half dq8 = nf5_data[q_1>>5 & 0x1f];
    half dq9 = nf5_data[q_1>>21 & 0x1f];
    half dq10 = nf5_data[q_1>>10 & 0x1f];
    half dq11 = nf5_data[q_1>>26 & 0x1f];
    half dq12 = nf5_data[q_2 & 0x1f];
    half dq13 = nf5_data[q_2>>16 & 0x1f];
    half dq14 = nf5_data[q_2>>5 & 0x1f];
    half dq15 = nf5_data[q_2>>21 & 0x1f];
    half dq16 = nf5_data[q_2>>10 & 0x1f];
    half dq17 = nf5_data[q_2>>26 & 0x1f];
    half dq18 = nf5_data[q_3 & 0x1f];
    half dq19 = nf5_data[q_3>>16 & 0x1f];
    half dq20 = nf5_data[q_3>>5 & 0x1f];
    half dq21 = nf5_data[q_3>>21 & 0x1f];
    half dq22 = nf5_data[q_3>>10 & 0x1f];
    half dq23 = nf5_data[q_3>>26 & 0x1f];
    half dq24 = nf5_data[q_4 & 0x1f];
    half dq25 = nf5_data[q_4>>16 & 0x1f];
    half dq26 = nf5_data[q_4>>5 & 0x1f];
    half dq27 = nf5_data[q_4>>21 & 0x1f];
    half dq28 = nf5_data[q_4>>10 & 0x1f];
    half dq29 = nf5_data[q_4>>26 & 0x1f];
    half dq30 = nf5_data[q_4>>10 & 0x1f];
    half dq31 = nf5_data[q_4>>26 & 0x1f];
    dq[0] = __halves2half2(dq0, dq1);
    dq[1] = __halves2half2(dq2, dq3);
    dq[2] = __halves2half2(dq4, dq5);
    dq[3] = __halves2half2(dq6, dq7);
    dq[4] = __halves2half2(dq8, dq9);
    dq[5] = __halves2half2(dq10, dq11);
    dq[6] = __halves2half2(dq12, dq13);
    dq[7] = __halves2half2(dq14, dq15);
    dq[8] = __halves2half2(dq16, dq17);
    dq[9] = __halves2half2(dq18, dq19);
    dq[10] = __halves2half2(dq20, dq21);
    dq[11] = __halves2half2(dq22, dq23);
    dq[12] = __halves2half2(dq24, dq25);
    dq[13] = __halves2half2(dq26, dq27);
    dq[14] = __halves2half2(dq28, dq29);
    dq[15] = __halves2half2(dq30, dq31);
}

#else

__forceinline__ __device__ void shuffle_5bit_32
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_5bit_32
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    const uint32_t q_3,
    const uint32_t q_4,
    half2 (&dq)[16],
    int stride
)
{
    half dqh[32];
    for (int i = 0; i <  6; i++) dqh[     i] = dq_ns(exb(     q_0, i * 5    , 0x1f), 16);
                                 dqh[ 6    ] = dq_ns(exb(q_1, q_0,        30, 0x1f), 16);
    for (int i = 0; i <  5; i++) dqh[ 7 + i] = dq_ns(exb(     q_1, i * 5 + 3, 0x1f), 16);
                                 dqh[12    ] = dq_ns(exb(q_2, q_1,        28, 0x1f), 16);
    for (int i = 0; i <  6; i++) dqh[13 + i] = dq_ns(exb(     q_2, i * 5 + 1, 0x1f), 16);
                                 dqh[19    ] = dq_ns(exb(q_3, q_2,        31, 0x1f), 16);
    for (int i = 0; i <  5; i++) dqh[20 + i] = dq_ns(exb(     q_3, i * 5 + 4, 0x1f), 16);
                                 dqh[25    ] = dq_ns(exb(q_4, q_3,        29, 0x1f), 16);
    for (int i = 0; i <  6; i++) dqh[26 + i] = dq_ns(exb(     q_4, i * 5 + 2, 0x1f), 16);

    for (int i = 0; i < 16; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif