#include <iostream>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <cstdint>
#include <cstring>


//生成预计算T-table
void generate_SM4_Ttable(uint32_t T0[256], uint32_t T1[256], uint32_t T2[256], uint32_t T3[256]) {
    for (int i = 0; i < 256; i++) {
        uint32_t s = SBOX[i];
        uint32_t val = L((s << 24) | (s << 16) | (s << 8) | s); 
        T0[i] = val;
        T1[i] = (val << 8 | val >> 24);
        T2[i] = (val << 16 | val >> 16);
        T3[i] = (val << 24 | val >> 8);
    }
}

//使用T-table优化轮函数，避免线性运算周期
__m128i SM4_round_T(__m128i x0,__m128i x1, __m128i x2, __m128i x3, uint32_t rk,const uint32_t T0[256], const uint32_t T1[256],const uint32_t T2[256], const uint32_t T3[256]) {
    __m128i tmp = _mm_xor_si128(_mm_xor_si128(x1, x2), _mm_xor_si128(x3, _mm_set1_epi32(rk)));
    alignas(16) uint32_t arr[4];
    _mm_store_si128((__m128i*)arr,tmp );
    uint32_t t = T0[(arr[0] >> 24) & 0xFF] ^ T1[(arr[1] >> 16) & 0xFF] ^
        T2[(arr[2] >> 8) & 0xFF] ^ T3[arr[3] & 0xFF];
    return _mm_xor_si128(x0, _mm_set1_epi32(t));



void SM4_ENC_SIMD_4_T(const uint32_t IN[4][4],uint32_t OUT[4][4], const uint32_t rk[32], const uint32_t T0[256], const uint32_t T1[256], const uint32_t T2[256], const uint32_t T3[256]) {
    __m128i x0 = _mm_set_epi32(IN[3][0], IN[2][0], IN[1][0], IN[0][0]);
    __m128i x1 = _mm_set_epi32(IN[3][1], IN[2][1], IN[1][1], IN[0][1]);
    __m128i x2 = _mm_set_epi32(IN[3][2], IN[2][2], IN[1][2], IN[0][2]);
    __m128i x3 = _mm_set_epi32(IN[3][3], IN[2][3], IN[1][3], IN[0][3]);

    for (int round = 0; round < 32; round++) {
        __m128i tmp = SM4_round_T(x0, x1, x2, x3, rk[round], T0, T1, T2, T3);
        x0 = x1; x1 = x2; x2 = x3; x3 = tmp;
    }

    x3 = _mm_set_epi32(
        _mm_extract_epi32(x3, 0),
        _mm_extract_epi32(x2, 0),
        _mm_extract_epi32(x1, 0),
        _mm_extract_epi32(x0, 0)
    );

    alignas(16) uint32_t result[4];
    _mm_store_si128((__m128i*)result, x3);
    for (int i = 0; i < 4; i++) memcpy(OUT[i], &result[i], 4);
}
