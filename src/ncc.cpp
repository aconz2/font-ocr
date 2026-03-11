#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>

#ifndef __builtin_assume
#define __builtin_assume(x) (void)(x);
#endif

struct Match {
    uint32_t x, y, w, h;
    float similarity;
};

static inline uint32_t ncc_sum_table_sum(uint32_t* s, size_t W, size_t x, size_t y, size_t w, size_t h) {
    auto at = [s, W](size_t x, size_t y) {
        return s[y * W + x];
    };
    auto a = at(x + w - 1, y + h - 1);
    auto b = at(x     - 1, y + h - 1);
    auto c = at(x + w - 1, y     - 1);
    auto d = at(x     - 1, y     - 1);
    return a + d - b - c;
}

static inline uint64_t ncc_sumsqr_table_sum(uint64_t* s, size_t W, size_t x, size_t y, size_t w, size_t h) {
    auto at = [s, W](size_t x, size_t y) {
        return s[y * W + x];
    };
    auto a = at(x + w - 1, y + h - 1);
    auto b = at(x     - 1, y + h - 1);
    auto c = at(x + w - 1, y     - 1);
    auto d = at(x     - 1, y     - 1);
    return a + d - b - c;
}

// https://stackoverflow.com/questions/23189488/horizontal-sum-of-32-bit-floats-in-256-bit-avx-vector
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

const size_t N = 8;

extern "C" size_t ncc_8_f32(
    float* __restrict reference,
    size_t r_w,
    size_t r_h,
    uint32_t* sum_table,
    uint64_t* sumsqr_table,
    uint8_t* needle,
    float* __restrict needle_f32, // this is N * n_h sized
    size_t n_w,
    size_t n_h,
    uint32_t* acc,
    float threshold,
    Match* out,
    size_t n_out
        ) {
    Match* out_cur = out;
    Match* out_fin = out + n_out;

    size_t n = n_w * n_h;
    size_t n_written = 0;

    uint32_t s_n = 0;
    uint32_t s2_n = 0;

    for (size_t i = 0; i < n_w * n_h; i++) {
        s_n += needle[i];
        s2_n += (uint32_t)needle[i] * (uint32_t)needle[i];
    }
    double norm2_n = (double)s2_n - (double)((uint64_t)s_n * (uint64_t)s_n) / (double)n;
    assert(s_n != 0);

    auto x_searches = r_w - N + 1;
    auto y_searches = r_h - N + 1;

    for (size_t y = 1; y < y_searches; y++) {
        for (size_t needle_y = 0; needle_y < n_h; needle_y++) {
            for (size_t x = 1; x < x_searches; x++) {
                auto r = &reference[(y + needle_y) * r_w + x];
                auto n = &needle_f32[needle_y * N];
                __m256 r_v = _mm256_loadu_ps(r);
                __m256 n_v = _mm256_loadu_ps(n);
                float acc_f = 0;
                __m256 v = _mm256_mul_ps(r_v, n_v);
                /*float tmp[N];*/
                /*_mm256_storeu_ps(tmp, v);*/
                /*for (size_t i = 0; i < N; i++) {*/
                /*    acc_f += tmp[i];*/
                /*}*/
                acc_f = _mm256_reduce_add_ps(v);
                /*float acc_f = 0;*/
                /*for (size_t i = 0; i < N; i++) {*/
                /*    acc_f += r[i] * n[i] * mask[i];*/
                /*}*/
                if (needle_y == 0) {
                    acc[x - 1] = acc_f;
                } else {
                    acc[x - 1] += acc_f;
                }
            }
        }

        for (size_t x = 1; x < x_searches; x++) {
            uint32_t acc_ = acc[x - 1];
            uint32_t s_p = ncc_sum_table_sum(sum_table, r_w, x, y, n_w, n_h);
            if (s_p == 0) {
                continue;
            }
            double num = (double)acc_ - (double)((uint64_t)s_n * (uint64_t)s_p) / (double)n;
            if (num < 0) {
                continue;
            }
            uint64_t s2_p = ncc_sumsqr_table_sum(sumsqr_table, r_w, x, y, n_w, n_h);
            double norm2_p = (double)s2_p - (double)((uint64_t)s_p * (uint64_t)s_p) / (double)n;
            double den = sqrt(norm2_n * norm2_p);
            float similarity = num / den;
            if (!(similarity >= -1.01 && similarity <= 1.01)) {
                fprintf(stderr, "bad similarity %.2f; x=%ld y=%ld norm_2n=%f norm_2p=%f acc=%d num=%f s_n=%d s_p=%d\n",
                        similarity, x, y, norm2_n, norm2_p, acc_, num, s_n, s_p
                        );
            }
            assert(similarity >= -1.01 && similarity <= 1.01);
            /*fprintf(stderr, "similarity %.2f\n", similarity);*/
            if (similarity > threshold) {
                *out_cur++ = {(uint32_t)x, (uint32_t)y, (uint32_t)n_w, (uint32_t)n_h, similarity};
                n_written += 1;
                if (out_cur == out_fin) {
                    return n_written;
                }
            }
        }
    }

    return n_written;
}

#define MANUAL_INTRIN

#ifdef MANUAL_INTRIN
static inline __m128i u16v_8_dot_v(__m128i a, __m128i b) {
    // a b c d e f g h
    // the madd gives us
    // 0  1  2  3
    // ab cd ef gh
    // ef gh ab cd + <- shuffle
    // abef cdgh efab ghcd
    // 0    1    2    3
    // abef cdgh efab ghcd + <- shuffle
    // cdgh abef ghcd efab
    __m128i x = _mm_madd_epi16(a, b);
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1)));
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));
    return x;
};
static inline uint32_t u16v_8_dot(__m128i a, __m128i b) {
    auto x = u16v_8_dot_v(a, b);
    return _mm_cvtsi128_si32(x);
}

static inline uint32_t u8_8_dot_u(uint8_t* a, uint8_t* b) {
    __m128i a_ = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)a));
    __m128i b_ = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)b));
    return u16v_8_dot(a_, b_);
};

static inline __m128i u8_8_dot_uv(uint8_t* a, uint8_t* b) {
    __m128i a_ = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)a));
    __m128i b_ = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)b));
    return u16v_8_dot_v(a_, b_);
};
#endif

extern "C" size_t ncc_8_u8(
    uint8_t* __restrict reference,
    size_t r_w,
    size_t r_h,
    uint8_t* needle,
    uint8_t* __restrict needle_u8, // this is N * n_h sized
    size_t n_w,
    size_t n_h,
    uint32_t* acc,
    uint32_t* patch_sum,
    double* patch_norm,
    uint16_t* start_end,
    float threshold,
    Match* out,
    size_t n_out
        ) {
    double threshold_d = threshold;

    Match* out_cur = out;
    Match* out_fin = out + n_out;

    size_t n = n_w * n_h;
    size_t n_written = 0;

    uint32_t s_n = 0;
    uint32_t s2_n = 0;

    for (size_t i = 0; i < n_w * n_h; i++) {
        s_n += needle[i];
        s2_n += (uint32_t)needle[i] * (uint32_t)needle[i];
    }
    double norm2_n = (double)s2_n - (double)((uint64_t)s_n * (uint64_t)s_n) / (double)n;
    double norm_n = sqrt(norm2_n);
    assert(s_n != 0);

    /*auto x_searches = r_w - N + 1;*/
    auto y_searches = r_h - N + 1;

// this is slower
//
#define BSLRI
/*#define QUAD*/
    // delayed sum good
/*#define DELAYED_SUM*/
/*#define DELAYED_SUM_ALIGNR*/

    for (size_t y = 1; y < y_searches; y++) {
        uint16_t start = start_end[y * 2 + 0];
        uint16_t end = start_end[y * 2 + 1];

#ifdef BSRLI
        auto inner = [&](size_t needle_y) {
            size_t x = start;
            __m128i windows[8];
            __m128i n = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&needle_u8[needle_y * N]));
            for (; x + N <= end; x += N) {
                // this loads 16 elements, then we can shift them over
                __m128i r1 = _mm_loadu_si128((__m128i*)&reference[(y + needle_y) * r_w + x]);

                windows[0] = r1;
                // I was using alignr here but realized I was loading 16 bytes anyways
                // so it became _mm_alignr_epi8(r1, r1, 1) and same as shifting
                windows[1] = _mm_bsrli_si128(r1, 1);
                windows[2] = _mm_bsrli_si128(r1, 2);
                windows[3] = _mm_bsrli_si128(r1, 3);
                windows[4] = _mm_bsrli_si128(r1, 4);
                windows[5] = _mm_bsrli_si128(r1, 5);
                windows[6] = _mm_bsrli_si128(r1, 6);
                windows[7] = _mm_bsrli_si128(r1, 7);
                if (needle_y == 0) {
                    for (size_t j = 0; j < 8; j++) {
                        acc[x + j] = u16v_8_dot(n, _mm_cvtepu8_epi16(windows[j]));
                    }
                } else {
                    for (size_t j = 0; j < 8; j++) {
                        acc[x + j] += u16v_8_dot(n, _mm_cvtepu8_epi16(windows[j]));
                    }
                }
            }
            for (; x < end; x += 1) {
                auto n = &needle_u8[needle_y * N];
                auto r = &reference[(y + needle_y) * r_w + x];
                uint32_t acc_ = u8_8_dot_u(n, r);
                if (needle_y == 0) {
                    acc[x] = acc_;
                } else {
                    acc[x] += acc_;
                }
            }
        };
#elif defined(DELAYED_SUM)
        auto inner = [&](size_t needle_y) {
            __m128i n = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&needle_u8[needle_y * N]));
            for (size_t x = start; x < end; x++) {
                __m128i r = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&reference[(y + needle_y) * r_w + x]));
                __m128i nr = _mm_madd_epi16(n, r); // 4 32 bit partial sums
                if (needle_y == 0) {
                    _mm_storeu_si128((__m128i*)&acc[x * 4], nr);
                } else {
                    __m128i a = _mm_loadu_si128((__m128i*)&acc[x * 4]);
                    a = _mm_add_epi32(a, nr);
                    _mm_storeu_si128((__m128i*)&acc[x * 4], a);
                }
            }
        };
#elif defined(DELAYED_SUM_ALIGNR)
        auto inner = [&](size_t needle_y) {
            __m128i windows[8];
            __m128i n = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&needle_u8[needle_y * N]));
            __m128i r1, r2;
            size_t x = start;
            if (start + 1 == end) { return; }
            r1 = _mm_loadu_si128((__m128i*)&reference[(y + needle_y) * r_w + x]);
            for (; x + (N * 2) <= end; x += N) {
                r2 = _mm_loadu_si128((__m128i*)&reference[(y + needle_y) * r_w + x + N]);

                windows[0] = r1;
                windows[1] = _mm_alignr_epi8(r2, r1, 1);
                windows[2] = _mm_alignr_epi8(r2, r1, 2);
                windows[3] = _mm_alignr_epi8(r2, r1, 3);
                windows[4] = _mm_alignr_epi8(r2, r1, 4);
                windows[5] = _mm_alignr_epi8(r2, r1, 5);
                windows[6] = _mm_alignr_epi8(r2, r1, 6);
                windows[7] = _mm_alignr_epi8(r2, r1, 7);
                if (needle_y == 0) {
                    for (size_t j = 0; j < 8; j++) {
                        __m128i v = _mm_madd_epi16(n, _mm_cvtepu8_epi16(windows[j]));
                        _mm_storeu_si128((__m128i*)&acc[(x + j) * 4], v);
                    }
                } else {
                    for (size_t j = 0; j < 8; j++) {
                        __m128i a = _mm_loadu_si128((__m128i*)&acc[(x + j) * 4]);
                        __m128i v = _mm_madd_epi16(n, _mm_cvtepu8_epi16(windows[j]));
                        a = _mm_add_epi32(a, v);
                        _mm_storeu_si128((__m128i*)&acc[(x + j) * 4], a);
                    }
                }
                r1 = r2;
            }
            for (; x < end; x++) {
                __m128i r = _mm_cvtepu8_epi16(_mm_loadu_si64((__m128i*)&reference[(y + needle_y) * r_w + x]));
                __m128i nr = _mm_madd_epi16(n, r); // 4 32 bit partial sums
                if (needle_y == 0) {
                    _mm_storeu_si128((__m128i*)&acc[x * 4], nr);
                } else {
                    __m128i a = _mm_loadu_si128((__m128i*)&acc[x * 4]);
                    a = _mm_add_epi32(a, nr);
                    _mm_storeu_si128((__m128i*)&acc[x * 4], a);
                }
            }
        };
#elif defined(QUAD)
        auto inner = [&](size_t needle_y) {
            auto n = &needle_u8[needle_y * N];
            size_t x = start;
            for (; x + 4 <= end; x += 4) {
                __m128i a = _mm_loadu_si128((__m128i*)&acc[x]);
                __m128i s1 = u8_8_dot_uv(n, &reference[(y + needle_y) * r_w + x + 0]);
                __m128i s2 = u8_8_dot_uv(n, &reference[(y + needle_y) * r_w + x + 1]);
                __m128i s3 = u8_8_dot_uv(n, &reference[(y + needle_y) * r_w + x + 2]);
                __m128i s4 = u8_8_dot_uv(n, &reference[(y + needle_y) * r_w + x + 3]);
#define BLEND
#ifdef BLEND
                // mask: 0 is first arg, 1 is second; read from right to left
                __m128i s12 = _mm_blend_epi32(s1, s2, 0b0010); // [s1, s2, s1, s1]
                __m128i s34 = _mm_blend_epi32(s3, s4, 0b1000); // [s3, s3, s3, s4]
                __m128i s = _mm_blend_epi32(s12, s34, 0b1100); // [s1, s2, s3, s4]
#else
                __m128i s12 = _mm_unpacklo_epi32(s1, s2); // [S1, S2, S1, S2]
                __m128i s34 = _mm_unpacklo_epi32(s3, s4); // [S3, S4, S3, S4]
                __m128i s = _mm_unpacklo_epi64(s12, s34); // [S1, S2, S3, S4]
#endif
                if (needle_y == 0) {
                    _mm_storeu_si128((__m128i*)&acc[x], s);
                } else {
                    a = _mm_add_epi32(a, s);
                    _mm_storeu_si128((__m128i*)&acc[x], a);
                }
            }
            for (; x < end; x += 1) {
                auto r = &reference[(y + needle_y) * r_w + x];
                uint32_t acc_ = u8_8_dot_u(n, r);
                if (needle_y == 0) {
                    acc[x] = acc_;
                } else {
                    acc[x] += acc_;
                }
            }
        };
#else
        auto inner = [&](size_t needle_y) {
            auto n = &needle_u8[needle_y * N];
            for (size_t x = start; x < end; x++) {
                auto r = &reference[(y + needle_y) * r_w + x];
#ifdef MANUAL_INTRIN
                uint32_t acc_ = u8_8_dot_u(n, r);
#else
                uint32_t acc_ = 0;
                for (size_t i = 0; i < N; i++) {
                    acc_ += r[i] * n[i];
                }
#endif
                if (needle_y == 0) {
                    acc[x] = acc_;
                } else {
                    acc[x] += acc_;
                }
            }
        };
#endif

        // doing two rows at once isn't faster
/*#define DOUBLE*/

#ifdef DOUBLE
        auto inner2 = [&](size_t needle_y) {
            for (size_t x = start; x < end; x++) {
                auto r1 = &reference[(y + needle_y) * r_w + x];
                auto r2 = &reference[(y + needle_y + 1) * r_w + x];
                auto n1 = &needle_u8[needle_y * N];
                auto n2 = &needle_u8[(needle_y + 1) * N];
                uint32_t acc_ = 0;
#ifdef MANUAL_INTRIN
                acc_ = u8_8_dot_u(r1, n1) + u8_8_dot_u(r2, n2);
#else
                for (size_t i = 0; i < N; i++) { acc_ += r1[i] * n1[i]; }
                for (size_t i = 0; i < N; i++) { acc_ += r2[i] * n2[i]; }
#endif
                if (needle_y == 0) {
                    acc[x] = acc_;
                } else {
                    acc[x] += acc_;
                }
            }
        };

        inner(0);
        size_t needle_y = 1;
        for (; needle_y + 1 < n_h; needle_y += 2) {
            inner2(needle_y);
        }
        for (; needle_y < n_h; needle_y++) {
            inner(needle_y);
        }
#else
        inner(0);
        for (size_t needle_y = 1; needle_y < n_h; needle_y++) {
            inner(needle_y);
        }
#endif

        // check num > threshold_d * den, not 100% sure this is numerically accurate
        // is maybe 1% faster
#define SCALED_COMPARE

        for (size_t x = start; x < end; x++) {
#ifdef DELAYED_SUM
            uint32_t acc_ = 0;
            for (size_t i = 0; i < 4; i ++) {
                acc_ += acc[(x * 4) + i];
            }
#else
            uint32_t acc_ = acc[x];
#endif
            uint32_t s_p = patch_sum[y * r_w + x];
            if (s_p == 0) {
                continue;
            }
            int64_t num_i64 = (int64_t)n * (int64_t)acc_ - ((int64_t)s_n * (int64_t)s_p);
            if (num_i64 < 0) {
                continue;
            }
            double num = (double)acc_ - (double)((uint64_t)s_n * (uint64_t)s_p) / (double)n;

            double norm_p = patch_norm[y * r_w + x];
            double den = norm_n * norm_p;
#ifdef SCALED_COMPARE
#else
            double similarity = num / den;
#endif
            /*if (!(similarity >= -1.01 && similarity <= 1.01)) {*/
            /*    fprintf(stderr, "bad similarity %.2f; x=%ld y=%ld norm_2n=%f norm_2p=%f acc=%d num=%f s_n=%d s_p=%d\n",*/
            /*            similarity, x, y, norm2_n, norm2_p, acc_, num, s_n, s_p*/
            /*            );*/
            /*}*/
            /*assert(similarity >= -1.01 && similarity <= 1.01);*/
#ifdef SCALED_COMPARE
            if (num > threshold_d * den) {
                *out_cur++ = {(uint32_t)x, (uint32_t)y, (uint32_t)n_w, (uint32_t)n_h, (float)(num / den)};
#else
            if (similarity > threshold_d) {
                *out_cur++ = {(uint32_t)x, (uint32_t)y, (uint32_t)n_w, (uint32_t)n_h, (float)similarity};
#endif
                n_written += 1;
                if (out_cur == out_fin) {
                    return n_written;
                }
            }
        }
    }

    return n_written;
}
