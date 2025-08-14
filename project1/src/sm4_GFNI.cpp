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
    const unsigned long long M = 0x0000000000000000ULL;
    const int                B = 0x00;
    return _mm_gf2p8affine_epi64_epi8(x, M_A1, B_A1);
}
static inline __m128i gf_affine_aes_2_sm4(){
    const unsigned long long M = 0x0000000000000000ULL;
    const int                B = 0x00;
    return _mm_gf2p8affine_epi64_epi8(x, M_A1, B_A1);
  
}
