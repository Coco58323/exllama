#ifndef _qdq_5_cuh
#define _qdq_5_cuh
#include <stdio.h>
#include "qdq_util.cuh"
#include "../../config.h"

// Permutation:
// 77775555 33331111  66664444 22220000
// ffffdddd bbbb9999  eeeecccc aaaa8888
// nnnnllll jjjjhhhh  mmmmkkkk iiiigggg
// vvvvtttt rrrrpppp  uuuussss qqqqoooo
// 02468ace gikmoqsu  13579bdfh jlnprtv

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

    uint32_t qq0 = qa & 0x0000000f;
    uint32_t qq1 = (qa>>5) & 0x0000000f;
    uint32_t qq2 = (qa>>10) & 0x0000000f;
    uint32_t qq3 = (qa>>15) & 0x0000000f;
    uint32_t qq4 = (qa>>20) & 0x0000000f;
    uint32_t qq5 = (qa>>25) & 0x0000000f;

    uint32_t qq6 = qb & 0x0000000f;
    uint32_t qq7 = (qb>>5) & 0x0000000f;
    uint32_t qq8 = (qb>>10) & 0x0000000f;
    uint32_t qq9 = (qb>>15) & 0x0000000f;
    uint32_t qqa = (qb>>20) & 0x0000000f;
    uint32_t qqb = (qb>>25) & 0x0000000f;

    uint32_t qqc = qc & 0x0000000f;
    uint32_t qqd = (qc>>5) & 0x0000000f;
    uint32_t qqe = (qc>>10) & 0x0000000f;
    uint32_t qqf = (qc>>15) & 0x0000000f;
    uint32_t qqg = (qc>>20) & 0x0000000f;
    uint32_t qqh = (qc>>25) & 0x0000000f;

    uint32_t qqi = qd & 0x0000000f;
    uint32_t qqj = (qd>>5) & 0x0000000f;
    uint32_t qqk = (qd>>10) & 0x0000000f;
    uint32_t qql = (qd>>15) & 0x0000000f;
    uint32_t qqm = (qd>>20) & 0x0000000f;
    uint32_t qqn = (qd>>25) & 0x0000000f;

    uint32_t qqo = qe & 0x0000000f;
    uint32_t qqp = (qe>>5) & 0x0000000f;
    uint32_t qqq = (qe>>10) & 0x0000000f;
    uint32_t qqr = (qe>>15) & 0x0000000f;
    uint32_t qqs = (qe>>20) & 0x0000000f;
    uint32_t qqt = (qe>>25) & 0x0000000f;

    uint32_t qqu = qf & 0x0000000f;
    uint32_t qqv = (qf>>5) & 0x0000000f;

    // uint32_t s0 = (qa >> 4) & 0x00000001;
    // uint32_t s1 = (qa >> 9) & 0x00000001;
    // uint32_t s2 = (qa >> 14) & 0x00000001;
    // uint32_t s3 = (qa >> 19) & 0x00000001;
    // uint32_t s4 = (qa >> 24) & 0x00000001;
    // uint32_t s5 = (qa >> 29) & 0x00000001;

    // uint32_t s6 = (qb >> 4) & 0x00000001;
    // uint32_t s7 = (qb >> 9) & 0x00000001;
    // uint32_t s8 = (qb >> 14) & 0x00000001;
    // uint32_t s9 = (qb >> 19) & 0x00000001;
    // uint32_t sa = (qb >> 24) & 0x00000001;
    // uint32_t sb = (qb >> 29) & 0x00000001;

    // uint32_t sc = (qc >> 4) & 0x00000001;
    // uint32_t sd = (qc >> 9) & 0x00000001;
    // uint32_t se = (qc >> 14) & 0x00000001;
    // uint32_t sf = (qc >> 19) & 0x00000001;
    // uint32_t sg = (qc >> 24) & 0x00000001;
    // uint32_t sh = (qc >> 29) & 0x00000001;

    // uint32_t si = (qd >> 4) & 0x00000001;
    // uint32_t sj = (qd >> 9) & 0x00000001;
    // uint32_t sk = (qd >> 14) & 0x00000001;
    // uint32_t sl = (qd >> 19) & 0x00000001;
    // uint32_t sm = (qd >> 24) & 0x00000001;
    // uint32_t sn = (qd >> 29) & 0x00000001;

    // uint32_t so = (qe >> 4) & 0x00000001;
    // uint32_t sp = (qe >> 9) & 0x00000001;
    // uint32_t sq = (qe >> 14) & 0x00000001;
    // uint32_t sr = (qe >> 19) & 0x00000001;
    // uint32_t ss = (qe >> 24) & 0x00000001;
    // uint32_t st = (qe >> 29) & 0x00000001;

    // uint32_t su = (qf >> 4) & 0x00000001;
    // uint32_t sv = (qf >> 9) & 0x00000001;


    za = qq0 | (qq2<<4) | (qq4<<8) | (qq6<<12) | (qq1<<16) | (qq3<<20) | (qq5<<24) | (qq7<<28);
    zb = qq8 | (qqa<<4) | (qqc<<8) | (qqe<<12) | (qq9<<16) | (qqb<<20) | (qqd<<24) | (qqf<<28);
    zc = qqg | (qqi<<4) | (qqk<<8) | (qqm<<12) | (qqh<<16) | (qqj<<20) | (qql<<24) | (qqn<<28);
    zd = qqo | (qqq<<4) | (qqs<<8) | (qqu<<12) | (qqp<<16) | (qqr<<20) | (qqt<<24) | (qqv<<28);
    ze = qqv | (qqt<<1)| (qqr<<2)| (qqp<<3)| (qqn<<4)| (qql<<5)| (qqj<<6)| (qqh<<7)| (qqf<<8)| (qqd<<9)| (qqb<<10)| (qq9<<11)| (qq7<<12)| (qq5<<13)| (qq3<<14)| (qq1<<15)| (qqu<<16)| (qqs<<17)| (qqq<<18)| (qqo<<19)| (qqm<<20)| (qqk<<21)| (qqi<<22)| (qqg<<23)| (qqe<<24)| (qqc<<25)| (qqa<<26)| (qq8<<27)| (qq6<<28)| (qq4<<29)| (qq2<<30)| (qq0<<31);


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
    int stride
)
{
    // 77775555 33331111  66664444 22220000
    // hhhhffff ddddbbbb  ggggeeee ccccaaaa
    // ppppnnnn lllljjjj  oooommmm kkkkiiii
    // xxxxvvvv ttttrrrr  wwwwuuuu ssssqqqq
    // 0248ace gikmoqsuw  1357bdfh jlnprtvx
    // uint32_t qq[5] = {q_0, q_1, q_2, q_3, q_4};
    // half2_uint32 resultq[16];
    // for (int i = 0; i < 4; i++) {
    //     half2_uint32 qq0 ((qq[i] & 0x000f000f) << 7); // (q[ 0], q[ 1])
    //     half2_uint32 qq1 ((qq[i] & 0x00f000f0) << 3); // (q[ 2], q[ 3]) 
    //     qq[i] >>= 1;
    //     half2_uint32 qq2 ((qq[i] & 0x07800780)); // (q[ 4], q[ 5])     
    //     qq[i] >>= 4;
    //     half2_uint32 qq3 ((qq[i] & 0x07800780)); // (q[ 6], q[ 7])
    //     resultq[i * 4].as_uint32 = qq0.as_uint32;
    //     resultq[i * 4 + 1].as_uint32 = qq1.as_uint32;
    //     resultq[i * 4 + 2].as_uint32 = qq2.as_uint32;
    //     resultq[i * 4 + 3].as_uint32 = qq3.as_uint32;
    // }
    for (int i = 0; i < 16; i++) {
        half2_uint32 sign (q_4 & 0xf800f800);
        half2_uint32 qq ((q_0 & 0x000f000f) << 7);
        // dq[i] = resultq[i].as_half2
        dq[i] = __hmul2(sign.as_half2, qq.as_half2);
        q_4 <<= 1;
    }
}

#endif