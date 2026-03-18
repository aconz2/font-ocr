#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

struct Match {
    uint16_t x, y;
    float similarity;
};

static inline __m128i u32_4_sum(__m128i v) {
    //  3  2  1  0
    //  d  c  b  a
    //  c  d  a  b  (2 3 0 1)
    // cd cd ab ab
    // ab ab cd cd  (1 0 3 2)
    //        abcd
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    return v;
}

// sum 4 u32 in each lane, sum gets put in each lane
static inline __m256i u32_4_2_sum(__m256i v) {
    v = _mm256_add_epi32(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    v = _mm256_add_epi32(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    return v;
}

static inline __m128i u32_8_sum(__m256i v) {
    __m128i a, b;
    v = u32_4_2_sum(v);
    a = _mm256_extractf128_si256(v, 0);
    b = _mm256_extractf128_si256(v, 1);
    return _mm_add_epi32(a, b);
}

static inline uintptr_t align_to(uintptr_t p, size_t N) {
    return (p + N - 1) & ~(N - 1);
}

// requirements:
//  acc has at least 32 extra bytes so we can align it
//  n_out >= 1
//  n_h >= 4
//
extern "C" size_t ncc_8_u8(
    uint8_t* reference,
    size_t r_w,
    size_t r_h,
    uint8_t* needle_u8, // this is N * n_h sized
    size_t n_w,
    size_t n_h,
    uint32_t* acc,
    size_t acc_len,
    uint32_t* patch_sum,
    double* patch_rnorm,
    uint16_t* start_end,
    float threshold,
    Match* out,
    size_t n_out
        ) {
    Match* out_cur = out;
    Match* out_fin = out + n_out;

    memset(acc, 0, sizeof(uint32_t) * acc_len);
    acc = (uint32_t*)align_to((uintptr_t)acc, 32);

    size_t n = n_w * n_h;
    auto y_searches = r_h - n_h + 1;

    uint32_t s_n = 0;
    uint32_t s2_n = 0;

    for (size_t i = 0; i < n_h; i++) {
        for (size_t j = 0; j < 8; j++) {
            s_n += needle_u8[i * 8 + j];
            s2_n += (uint32_t)needle_u8[i * 8 + j] * (uint32_t)needle_u8[i * 8 + j];
        }
    }

    double threshold_d = threshold;
    double norm2_n = (double)s2_n - (double)((uint64_t)s_n * (uint64_t)s_n) / (double)n;
    double rnorm_n = 1. / sqrt(norm2_n);
    double n_recip = 1. / (double)n;

    __m256d vs_n = _mm256_set1_pd((double)s_n); // sum of needle double
    __m256d vn_recip = _mm256_set1_pd(n_recip); // n recip
    __m256d vrnorm_n = _mm256_set1_pd(rnorm_n); // rnorm_n
    __m256d vthreshold = _mm256_set1_pd(threshold);
    __m256d vinf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256i zero256 = _mm256_set1_epi8(0);
    __m128i zero128 = _mm_set1_epi8(0);

    const size_t B = 4;

    auto processv = [&out_cur, out_fin, vs_n, vn_recip, vrnorm_n, vthreshold, vinf](__m256d vacc, __m256d vs_p, __m256d vrn_p, size_t x, size_t y) {
        // acc - s_p * s_n * (1 / n)
        /*__m256d num = _mm256_sub_pd(vacc, _mm256_mul_pd(_mm256_mul_pd(vs_n, vs_p), vn_recip));*/
        __m256d num = _mm256_fnmadd_pd(_mm256_mul_pd(vs_n, vs_p), vn_recip, vacc);
        // (1 / norm_n) * (1 / norm_patch)
        __m256d den = _mm256_mul_pd(vrnorm_n, vrn_p);
        __m256d sim = _mm256_mul_pd(num, den);
        int mask = _mm256_movemask_pd(
                    _mm256_andnot_pd(
                        _mm256_cmp_pd(sim, vinf, _CMP_EQ_OQ),
                        _mm256_cmp_pd(sim, vthreshold, _CMP_GT_OQ)
                    ));
        if (mask == 0) return false; // fast path for no hits
        for (size_t i = 0; i < 4; i++) {
            if ((1 << i) & mask) {
                float sim_ = sim[i]; // TIL you can index into the lane
                *out_cur++ = {(uint16_t)(x + i), (uint16_t)y, sim_};
                if (out_cur == out_fin) {
                    return true;
                }
            }
        }
        return false;
    };

    for (size_t y = 1; y < y_searches; y++) {
        const uint16_t start = start_end[y * 2 + 0];
        const uint16_t end = start_end[y * 2 + 1];

        // needle gets replicated across both halfs so we can search at x and x+8 at the same time
        // acc gets pairs of 4 partial sums from (x, x+8) in a row, then the tail handling is as normal
        // on the last row of the needle, we accumulate the sums

        size_t needle_y = 0;
        // 4 rows of needle at a time, dual wide then single
        for (; needle_y + B <= n_h; needle_y += B) {
            __m128i n[B];
            __m256i nn[B];
            for (size_t i = 0; i < B; i++) {
                __m128i n_8 = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&needle_u8[(needle_y + i) * 8]));
                n[i] = n_8;
                nn[i] = _mm256_set_m128i(n_8, n_8);
            }
            size_t x = start;
            uint32_t* a = acc;
            for (; x + (8 * 2) < end; x += (8 * 2)) {
                for (size_t j = 0; j < 8; j++) {
                    __m256i nr[B];
                    for (size_t i = 0; i < B; i++) {
                        __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&reference[(y + needle_y + i) * r_w + x + j]));
                        nr[i] = _mm256_madd_epi16(nn[i], r); // 8 32 bit partial sums
                    }
                    __m256i s = _mm256_add_epi32(_mm256_add_epi32(nr[0], nr[1]), _mm256_add_epi32(nr[2], nr[3]));
                    s = _mm256_add_epi32(s, _mm256_load_si256((__m256i*)a));
                    _mm256_store_si256((__m256i*)a, s);
                    a += 8;
                }
            }
            for (; x < end; x++) {
                __m128i nr[B];
                for (size_t i = 0; i < B; i++) {
                    __m128i r = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&reference[(y + needle_y + i) * r_w + x]));
                    nr[i] = _mm_madd_epi16(n[i], r);
                }
                __m128i s = _mm_add_epi32(_mm_add_epi32(nr[0], nr[1]), _mm_add_epi32(nr[2], nr[3]));
                s = _mm_add_epi32(s, _mm_load_si128((__m128i*)a));
                _mm_store_si128((__m128i*)a, s);
                a += 4;
            }
        }

        // single rows of needle, dual wide then single
        for (; needle_y < n_h; needle_y++) {
            __m128i n_8 = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&needle_u8[needle_y * 8]));
            __m256i n = _mm256_set_m128i(n_8, n_8);
            size_t x = start;
            uint32_t* a = acc;
            for (; x + (8 * 2) < end; x += (8 * 2)) {
                for (size_t j = 0; j < 8; j++) {
                    __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&reference[(y + needle_y) * r_w + x + j]));
                    __m256i nr = _mm256_madd_epi16(n, r); // 8 32 bit partial sums
                    nr = _mm256_add_epi32(nr, _mm256_load_si256((__m256i*)a));
                    _mm256_store_si256((__m256i*)a, nr);
                    a += 8;
                }
            }
            for (; x < end; x++) {
                __m128i r = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&reference[(y + needle_y) * r_w + x]));
                __m128i nr = _mm_madd_epi16(n_8, r); // 4 32 bit partial sums
                nr = _mm_add_epi32(nr, _mm_load_si128((__m128i*)a));
                _mm_store_si128((__m128i*)a, nr);
                a += 4;
            }
        }

        size_t x = start;
        uint32_t* a = acc;
        for (; x + (8 * 2) < end; x += (8 * 2)) {
            __m256i A[8];
            // A holds acc sums in lo-hi lane pairs for 0,8 1,9 2,10 ... 7,15
            for (size_t i = 0; i < 8; i++) {
                A[i] = u32_4_2_sum(_mm256_load_si256((__m256i*)a));
                _mm256_store_si256((__m256i*)a, zero256);
                a += 8;
            }

            // we want to shuffle these into 4 32bit sums to then expand into doubles in order
            // (these diagrams are mirrored from lane layout)
            // each digit is the 32 bit sum
            // 0000,8888 1111,9999 2222,AAAA 3333,BBBB -> 0123,xxxx 89AB,xxxx -> 0x1x,2x3x 8x9x,AxBx
            //
            // 0000,8888
            // 1111,9999 unpacklo_epi32
            // 0101,8989
            // 2323,ABAB from other unpacklo_epi32
            // 0123,89AB blend
            // 0d1d,2d3d 8d9d,AdBd extract lanes and convert (d is the double)
            auto unpack = [](__m256d* v0123_d, __m256d* v89AB_d, __m256i v08, __m256i v19, __m256i v2A, __m256i v3B) {
                __m256i v0189 = _mm256_unpacklo_epi32(v08, v19);
                __m256i v23AB = _mm256_unpacklo_epi32(v2A, v3B);
                __m256i v0123_89AB = _mm256_blend_epi32(v0189, v23AB, 0b11001100);
                // NOTE we this treats our unsigned acc sum as signed, but we only use up to maybe 24 bits for 20x20 patches
                *v0123_d = _mm256_cvtepi32_pd(_mm256_extractf128_si256(v0123_89AB, 0));
                *v89AB_d = _mm256_cvtepi32_pd(_mm256_extractf128_si256(v0123_89AB, 1));
            };

            __m256d vacc[4];
            //                         08    19    2A    3B
            unpack(&vacc[0], &vacc[2], A[0], A[1], A[2], A[3]);
            //                         4C    5D    6E    7F
            unpack(&vacc[1], &vacc[3], A[4], A[5], A[6], A[7]);

            // compute the similarity for each batch of 4
            for (size_t i = 0; i < 4; i++) {
                __m256d v_s_p = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&patch_sum[y * r_w + x + i * 4]));
                __m256d v_rn_p = _mm256_loadu_pd(&patch_rnorm[y * r_w + x + i * 4]);
                if (processv(vacc[i], v_s_p, v_rn_p, x + i * 4, y)) {
                    return n_out;
                }
            }
        }

        for (; x < end; x++) {
            // acc is really 4 wide but we already summed on the last loop
            uint32_t acc_ = _mm_extract_epi32(u32_4_sum(_mm_load_si128((__m128i*)a)), 0);
            _mm_store_si128((__m128i*)a, zero128);
            uint32_t s_p = patch_sum[y * r_w + x];
            double num = (double)acc_ - (double)((uint64_t)s_n * (uint64_t)s_p) * n_recip;
            double den = rnorm_n * patch_rnorm[y * r_w + x];
            double similarity = num * den;
            if (similarity > threshold_d) {
                *out_cur++ = {(uint16_t)x, (uint16_t)y, (float)similarity};
                return n_out;
            }
            a += 4;
        }
    }

    return out_cur - out;
}

extern "C" size_t ncc_16_u8(
    uint8_t* reference,
    size_t r_w,
    size_t r_h,
    uint8_t* needle_u8, // this is N * n_h sized
    size_t n_w,
    size_t n_h,
    uint32_t* acc,
    size_t acc_len,
    uint32_t* patch_sum,
    double* patch_rnorm,
    uint16_t* start_end,
    float threshold,
    Match* out,
    size_t n_out
        ) {
    Match* out_cur = out;
    Match* out_fin = out + n_out;

    memset(acc, 0, sizeof(uint32_t) * acc_len);
    acc = (uint32_t*)align_to((uintptr_t)acc, 32);

    size_t n = n_w * n_h;
    auto y_searches = r_h - n_h + 1;

    uint32_t s_n = 0;
    uint32_t s2_n = 0;

    for (size_t i = 0; i < n_h; i++) {
        for (size_t j = 0; j < 16; j++) {
            s_n += needle_u8[i * 16 + j];
            s2_n += (uint32_t)needle_u8[i * 16 + j] * (uint32_t)needle_u8[i * 16 + j];
        }
    }

    double threshold_d = threshold;
    double norm2_n = (double)s2_n - (double)((uint64_t)s_n * (uint64_t)s_n) / (double)n;
    double rnorm_n = 1. / sqrt(norm2_n);
    double n_recip = 1. / (double)n;

    __m256d vs_n = _mm256_set1_pd((double)s_n); // sum of needle double
    __m256d vn_recip = _mm256_set1_pd(n_recip); // n recip
    __m256d vrnorm_n = _mm256_set1_pd(rnorm_n); // rnorm_n
    __m256d vthreshold = _mm256_set1_pd(threshold);

    auto processv = [&out_cur, out_fin, vs_n, vn_recip, vrnorm_n, vthreshold](__m256d vacc, __m256d vs_p, __m256d vrn_p, size_t x, size_t y) {
        // acc - s_p * s_n * (1 / n)
        /*__m256d num = _mm256_sub_pd(vacc, _mm256_mul_pd(_mm256_mul_pd(vs_n, vs_p), vn_recip));*/
        __m256d num = _mm256_fnmadd_pd(_mm256_mul_pd(vs_n, vs_p), vn_recip, vacc);
        // (1 / norm_n) * (1 / norm_patch)
        __m256d den = _mm256_mul_pd(vrnorm_n, vrn_p);
        __m256d sim = _mm256_mul_pd(num, den);
        int mask = _mm256_movemask_pd(_mm256_cmp_pd(sim, vthreshold, _CMP_GT_OQ));
        if (mask == 0) return false; // fast path for no hits
        for (size_t i = 0; i < 4; i++) {
            if ((1 << i) & mask) {
                float sim_ = sim[i]; // TIL you can index into the lane
                *out_cur++ = {(uint16_t)(x + i), (uint16_t)y, sim_};
                if (out_cur == out_fin) {
                    return true;
                }
            }
        }
        return false;
    };

    const size_t B = 4;

    for (size_t y = 1; y < y_searches; y++) {
        const uint16_t start = start_end[y * 2 + 0];
        const uint16_t end = start_end[y * 2 + 1];

        size_t needle_y = 0;
        for (; needle_y + B < n_h; needle_y += B) {
            __m256i n[B];
            for (size_t i = 0; i < B; i++) {
                n[i] = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&needle_u8[(needle_y + i) * 16]));
            }
            size_t acc_i = 0;
            for (size_t x = start; x < end; x++) {
                __m256i nr[B];
                for (size_t i = 0; i < B; i++) {
                    __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&reference[(y + needle_y + i) * r_w + x]));
                    nr[i] = _mm256_madd_epi16(n[i], r); // 8 32 bit partial sums
                }
                __m256i s = _mm256_add_epi32(_mm256_add_epi32(nr[0], nr[1]), _mm256_add_epi32(nr[2], nr[3]));
                auto a = acc + acc_i;
                if (needle_y == 0) {
                    _mm256_store_si256((__m256i*)a, s);
                } else if (needle_y == n_h - 1) {
                    s = _mm256_add_epi32(s, _mm256_load_si256((__m256i*)a));
                    __m128i s128 = u32_8_sum(s);
                    _mm_store_si128((__m128i*)a, s128);
                } else {
                    s = _mm256_add_epi32(s, _mm256_load_si256((__m256i*)a));
                    _mm256_store_si256((__m256i*)a, s);
                }
                acc_i += 8;
            }
        }
        for (; needle_y < n_h; needle_y++) {
            __m256i n_16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&needle_u8[needle_y * 16]));
            size_t acc_i = 0;
            for (size_t x = start; x < end; x++) {
                __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&reference[(y + needle_y) * r_w + x]));
                __m256i nr = _mm256_madd_epi16(n_16, r); // 8 32 bit partial sums
                auto a = acc + acc_i;
                if (needle_y == 0) {
                    _mm256_store_si256((__m256i*)a, nr);
                } else if (needle_y == n_h - 1) {
                    nr = _mm256_add_epi32(nr, _mm256_load_si256((__m256i*)a));
                    __m128i s = u32_8_sum(nr);
                    _mm_store_si128((__m128i*)a, s);
                } else {
                    nr = _mm256_add_epi32(nr, _mm256_load_si256((__m256i*)a));
                    _mm256_store_si256((__m256i*)a, nr);
                }
                acc_i += 8;
            }
        }

        size_t x = start;
        size_t acc_i = 0;
        for (; x + 4 < end; x += 4) {
            // last loop trip above sums the 8 lanes down into the first 4 entries
            __m128i vacc = _mm_set_epi32(acc[acc_i + 24], acc[acc_i + 16], acc[acc_i + 8], acc[acc_i]);
            acc_i += 4 * 8;
            __m256d vacc_d =_mm256_cvtepi32_pd(vacc);
            __m256d v_s_p = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&patch_sum[y * r_w + x]));
            __m256d v_rn_p = _mm256_loadu_pd(&patch_rnorm[y * r_w + x]);
            if (processv(vacc_d, v_s_p, v_rn_p, x, y)) {
                return n_out;
            }
        }
        for (; x < end; x++) {
            uint32_t acc_ = 0;
            for (size_t i = 0; i < 8; i ++) {
                acc_ += acc[acc_i + i];
            }
            uint32_t s_p = patch_sum[y * r_w + x];
            double num = (double)acc_ - (double)((uint64_t)s_n * (uint64_t)s_p) * n_recip;
            double den = rnorm_n * patch_rnorm[y * r_w + x];
            double similarity = num * den;
            if (similarity > threshold_d) {
                *out_cur++ = {(uint16_t)x, (uint16_t)y, (float)similarity};
                return n_out;
            }
            acc_i += 4;
        }
    }

    return out_cur - out;
}

// TOOD idea would be to use _mm_maddubs_epi16 with 7bit values but the difference might be too great
extern "C" size_t ncc_16_u8_7b(
    uint8_t* reference,
    size_t r_w,
    size_t r_h,
    uint8_t* needle_u8, // this is N * n_h sized
    size_t n_w,
    size_t n_h,
    uint32_t* acc,
    size_t acc_len,
    uint32_t* patch_sum,
    double* patch_rnorm,
    uint16_t* start_end,
    float threshold,
    Match* out,
    size_t n_out
        ) {
    Match* out_cur = out;
    Match* out_fin = out + n_out;

    memset(acc, 0, sizeof(uint32_t) * acc_len);
    acc = (uint32_t*)align_to((uintptr_t)acc, 32);

    size_t n = n_w * n_h;
    auto y_searches = r_h - n_h + 1;

    uint32_t s_n = 0;
    uint32_t s2_n = 0;

    for (size_t i = 0; i < n_h; i++) {
        for (size_t j = 0; j < 16; j++) {
            s_n += needle_u8[i * 16 + j];
            s2_n += (uint32_t)needle_u8[i * 16 + j] * (uint32_t)needle_u8[i * 16 + j];
        }
    }

    double threshold_d = threshold;
    double norm2_n = (double)s2_n - (double)((uint64_t)s_n * (uint64_t)s_n) / (double)n;
    double rnorm_n = 1. / sqrt(norm2_n);
    double n_recip = 1. / (double)n;

    __m256d vs_n = _mm256_set1_pd((double)s_n); // sum of needle double
    __m256d vn_recip = _mm256_set1_pd(n_recip); // n recip
    __m256d vrnorm_n = _mm256_set1_pd(rnorm_n); // rnorm_n
    __m256d vthreshold = _mm256_set1_pd(threshold);

    auto processv = [&out_cur, out_fin, vs_n, vn_recip, vrnorm_n, vthreshold](__m256d vacc, __m256d vs_p, __m256d vrn_p, size_t x, size_t y) {
        // acc - s_p * s_n * (1 / n)
        /*__m256d num = _mm256_sub_pd(vacc, _mm256_mul_pd(_mm256_mul_pd(vs_n, vs_p), vn_recip));*/
        __m256d num = _mm256_fnmadd_pd(_mm256_mul_pd(vs_n, vs_p), vn_recip, vacc);
        // (1 / norm_n) * (1 / norm_patch)
        __m256d den = _mm256_mul_pd(vrnorm_n, vrn_p);
        __m256d sim = _mm256_mul_pd(num, den);
        int mask = _mm256_movemask_pd(_mm256_cmp_pd(sim, vthreshold, _CMP_GT_OQ));
        if (mask == 0) return false; // fast path for no hits
        for (size_t i = 0; i < 4; i++) {
            if ((1 << i) & mask) {
                float sim_ = sim[i]; // TIL you can index into the lane
                *out_cur++ = {(uint16_t)(x + i), (uint16_t)y, sim_};
                if (out_cur == out_fin) {
                    return true;
                }
            }
        }
        return false;
    };

    // UBS is actually slower
/*#define UBS*/

    for (size_t y = 1; y < y_searches; y++) {
        const uint16_t start = start_end[y * 2 + 0];
        const uint16_t end = start_end[y * 2 + 1];

        for (size_t needle_y = 0; needle_y < n_h; needle_y++) {
#ifdef UBS
            __m128i n_16 = _mm_loadu_si128((__m128i*)&needle_u8[needle_y * 16]);
#else
            __m256i n_16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&needle_u8[needle_y * 16]));
#endif
            size_t acc_i = 0;
            for (size_t x = start; x < end; x++) {
#ifdef UBS
                __m128i r = _mm_loadu_si128((__m128i*)&reference[(y + needle_y) * r_w + x]);
                __m256i nr = _mm256_cvtepi16_epi32(_mm_maddubs_epi16(n_16, r));
#else
                __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)&reference[(y + needle_y) * r_w + x]));
                __m256i nr = _mm256_madd_epi16(n_16, r); // 8 32 bit partial sums
#endif
                auto a = acc + acc_i;
                if (needle_y == 0) {
                    _mm256_store_si256((__m256i*)a, nr);
                } else if (needle_y == n_h - 1) {
                    nr = _mm256_add_epi32(nr, _mm256_load_si256((__m256i*)a));
                    __m128i s = u32_8_sum(nr);
                    _mm_store_si128((__m128i*)a, s);
                } else {
                    nr = _mm256_add_epi32(nr, _mm256_load_si256((__m256i*)a));
                    _mm256_store_si256((__m256i*)a, nr);
                }
                acc_i += 8;
            }
        }

        size_t x = start;
        size_t acc_i = 0;
        for (; x + 4 < end; x += 4) {
            // last loop trip above sums the 8 lanes down into the first 4 entries
            __m128i vacc = _mm_set_epi32(acc[acc_i + 24], acc[acc_i + 16], acc[acc_i + 8], acc[acc_i]);
            acc_i += 4 * 8;
            __m256d vacc_d =_mm256_cvtepi32_pd(vacc);
            __m256d v_s_p = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&patch_sum[y * r_w + x]));
            __m256d v_rn_p = _mm256_loadu_pd(&patch_rnorm[y * r_w + x]);
            if (processv(vacc_d, v_s_p, v_rn_p, x, y)) {
                return n_out;
            }
        }
        for (; x < end; x++) {
            uint32_t acc_ = 0;
            for (size_t i = 0; i < 8; i ++) {
                acc_ += acc[acc_i + i];
            }
            uint32_t s_p = patch_sum[y * r_w + x];
            double num = (double)acc_ - (double)((uint64_t)s_n * (uint64_t)s_p) * n_recip;
            double den = rnorm_n * patch_rnorm[y * r_w + x];
            double similarity = num * den;
            if (similarity > threshold_d) {
                *out_cur++ = {(uint16_t)x, (uint16_t)y, (float)similarity};
                return n_out;
            }
            acc_i += 4;
        }
    }

    return out_cur - out;
}
