#include <immintrin.h>
#include <smmintrin.h>
#include <cstdint>
#include <cstring>

//32bit下的L变换
static inline uint32_t L32(uint32_t x) {
    return x ^ (x << 2 | x >> 30) ^ (x << 10 | x >> 22)
             ^ (x << 18 | x >> 14) ^ (x << 24 | x >> 8);
}

//使用GFNI指令计算AES域下的逆元
static inline __m128i gf_inv_aesfield(__m128i x) {
    __m128i zero = _mm_setzero_si128();
    __m128i is_zero = _mm_cmpeq_epi8(x, zero);
    __m128i inv = _mm_gf2p8inv_epi8(x);
    return _mm_andnot_si128(is_zero, inv);
}

//sm4域中元素与aes域中元素的仿射变换
static inline __m128i gf_affine_sm4_2_aes(){
    const unsigned long long M = 0x0673E556A2490912;
    const int                B = 0x01;
    return _mm_gf2p8affine_epi64_epi8(x, M_A1, B_A1);
}
static inline __m128i gf_affine_aes_2_sm4(){
    const unsigned long long M = 0x75E9E8356997606E;
    const int                B = 0x75;
    return _mm_gf2p8affine_epi64_epi8(x, M_A1, B_A1);
  
}
//在aes域上进行s盒查找
static inline __m128i Sbox_GNFI(__m128i in_bytes) {
    // 映射到AES域
    __m128i t = gf_affine_sm4_2_aes(in_bytes);
    
    // AES 域中求逆
    t = gf_inv_aesfield(t);
    
    // 通过逆映射还原到SM4域
    t = _mm_gf2p8affine_epi64_epi8(t);
    return t;
}

//GNFI版本的轮函数
static inline __m128i SM4_round_GFNI(__m128i x0, __m128i x1, __m128i x2, __m128i x3, uint32_t rk) {
    __m128i t = sm4_sbox_gfni(_mm_xor_si128(_mm_xor_si128(x1, x2), _mm_xor_si128(x3, _mm_set1_epi32(rk))));
    alignas(16) uint32_t tmp[4];
    _mm_store_si128((__m128i*)tmp, t);
    tmp[0] = L32(tmp[0]); tmp[1] = L32(t[1]);
    tmp[2] = L32(tmp[2]); tmp[3] = L32(t[3]);
    __m128i res=_mm_load_si128((__m128i*)tmp);
    return _mm_xor_si128(x0, res);
}

void SM4_ENC_SIMD_4_GFNI(const uint32_t IN[4][4], uint32_t OUT[4][4], const uint32_t rk[32]) {
    __m128i x0 = _mm_set_epi32(IN[3][0], IN[2][0], IN[1][0], IN[0][0]);
    __m128i x1 = _mm_set_epi32(IN[3][1], IN[2][1], IN[1][1], IN[0][1]);
    __m128i x2 = _mm_set_epi32(IN[3][2], IN[2][2], IN[1][2], IN[0][2]);
    __m128i x3 = _mm_set_epi32(IN[3][3], IN[2][3], IN[1][3], IN[0][3]);

    for (int round = 0; round < 32; ++round) {
        __m128i tmp = SM4_round_GFNI(x0, x1, x2, x3, rk[round]);
        x0 = x1; x1 = x2; x2 = x3; x3 = tmp;
    }

    alignas(16) uint32_t result[4];
    _mm_store_si128((__m128i*)result, x3);
    for (int i = 0; i < 4; ++i) {
        std::memcpy(OUT[i], &result[i], 4);
    }
}
