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
#pragma nounroll
#pragma clang loop vectorize(disable)
    for (size_t i = 0; i < n_w * n_h; i++) {
        s_n += needle[i];
        s2_n += (uint32_t)needle[i] * (uint32_t)needle[i];
    }
    double norm2_n = (double)s2_n - (double)((uint64_t)s_n * (uint64_t)s_n) / (double)n;
    assert(s_n != 0);

    auto x_searches = r_w - N + 1;
    auto y_searches = r_h - N + 1;

#pragma nounroll
#pragma clang loop vectorize(disable)
    for (size_t y = 1; y < y_searches; y++) {
#pragma nounroll
#pragma clang loop vectorize(disable)
        for (size_t needle_y = 0; needle_y < n_h; needle_y++) {
/*#pragma nounroll*/
#pragma clang loop vectorize(disable)
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

/*#pragma nounroll*/
#pragma clang loop vectorize(disable)
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

extern "C" size_t ncc_8_u8(
    uint8_t* __restrict reference,
    size_t r_w,
    size_t r_h,
    uint32_t* sum_table,
    uint64_t* sumsqr_table,
    uint8_t* needle,
    uint8_t* __restrict needle_u8, // this is N * n_h sized
    size_t n_w,
    size_t n_h,
    uint32_t* acc,
    uint32_t* patch_sum,
    double* patch_norm,
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
        auto inner = [&](size_t needle_y) {
            for (size_t x = 1; x < x_searches; x++) {
                auto r = &reference[(y + needle_y) * r_w + x];
                auto n = &needle_u8[needle_y * N];
                uint32_t acc_ = 0;
                for (size_t i = 0; i < N; i++) {
                    acc_ += (uint16_t)r[i] * (uint16_t)n[i];
                }
                if (needle_y == 0) {
                    acc[x - 1] = acc_;
                } else {
                    acc[x - 1] += acc_;
                }
            }
        };
        inner(0);
        for (size_t needle_y = 1; needle_y < n_h; needle_y++) {
            inner(needle_y);
        }

        for (size_t x = 1; x < x_searches; x++) {
            uint32_t acc_ = acc[x - 1];
            /*uint32_t s_p = ncc_sum_table_sum(sum_table, r_w, x, y, n_w, n_h);*/
            uint32_t s_p = patch_sum[y * r_w + x];
            if (s_p == 0) {
                continue;
            }
            int64_t num_i64 = (int64_t)n * (int64_t)acc_ - ((int64_t)s_n * (int64_t)s_p);
            if (num_i64 < 0) {
                continue;
            }
            double num = (double)acc_ - (double)((uint64_t)s_n * (uint64_t)s_p) / (double)n;
            /*if (num < 0) {*/
            /*    continue;*/
            /*}*/
            /*uint64_t s2_p = ncc_sumsqr_table_sum(sumsqr_table, r_w, x, y, n_w, n_h);*/
            /*double norm2_p = (double)s2_p - (double)((uint64_t)s_p * (uint64_t)s_p) / (double)n;*/
            double norm2_p = patch_norm[y * r_w + x];
            double den = sqrt(norm2_n * norm2_p);
            double similarity = num / den;
            /*if (!(similarity >= -1.01 && similarity <= 1.01)) {*/
            /*    fprintf(stderr, "bad similarity %.2f; x=%ld y=%ld norm_2n=%f norm_2p=%f acc=%d num=%f s_n=%d s_p=%d\n",*/
            /*            similarity, x, y, norm2_n, norm2_p, acc_, num, s_n, s_p*/
            /*            );*/
            /*}*/
            /*assert(similarity >= -1.01 && similarity <= 1.01);*/
            /*fprintf(stderr, "similarity %.2f\n", similarity);*/
            if (similarity > threshold) {
                *out_cur++ = {(uint32_t)x, (uint32_t)y, (uint32_t)n_w, (uint32_t)n_h, (float)similarity};
                n_written += 1;
                if (out_cur == out_fin) {
                    return n_written;
                }
            }
        }
    }

    return n_written;
}

extern "C" size_t ncc_8_u8_b2(
    uint8_t* __restrict reference,
    size_t r_w,
    size_t r_h,
    uint32_t* sum_table,
    uint64_t* sumsqr_table,
    uint8_t* needle_1,
    uint8_t* needle_2,
    uint8_t* __restrict needle_u8_1, // this is N * n_h sized
    uint8_t* __restrict needle_u8_2, // this is N * n_h sized
    size_t n_w,
    size_t n_h,
    uint32_t* acc,
    uint32_t* patch_sum,
    double* patch_norm,
    float threshold,
    Match* out,
    size_t n_out
        ) {

    Match* out_cur = out;
    Match* out_fin = out + n_out;

    size_t n = n_w * n_h;
    size_t n_written = 0;

    uint32_t s_n_1 = 0;
    uint32_t s2_n_1 = 0;
    uint32_t s_n_2 = 0;
    uint32_t s2_n_2 = 0;

    for (size_t i = 0; i < n_w * n_h; i++) {
        s_n_1 += needle_1[i];
        s_n_2 += needle_2[i];
        s2_n_1 += (uint32_t)needle_1[i] * (uint32_t)needle_1[i];
        s2_n_2 += (uint32_t)needle_2[i] * (uint32_t)needle_2[i];
    }
    double norm2_n_1 = (double)s2_n_1 - (double)((uint64_t)s_n_1 * (uint64_t)s_n_1) / (double)n;
    double norm2_n_2 = (double)s2_n_2 - (double)((uint64_t)s_n_2 * (uint64_t)s_n_2) / (double)n;

    auto x_searches = r_w - N + 1;
    auto y_searches = r_h - N + 1;

    for (size_t y = 1; y < y_searches; y++) {
        auto inner = [&](size_t needle_y) {
            auto n1 = &needle_u8_1[needle_y * N];
            auto n2 = &needle_u8_2[needle_y * N];
            for (size_t x = 1; x < x_searches; x++) {
                auto r = &reference[(y + needle_y) * r_w + x];
                uint32_t acc_1 = 0;
                uint32_t acc_2 = 0;
                for (size_t i = 0; i < N; i++) {
                    acc_1 += (uint16_t)r[i] * (uint16_t)n1[i];
                    acc_2 += (uint16_t)r[i] * (uint16_t)n2[i];
                }
                if (needle_y == 0) {
                    acc[(x - 1) * 2 + 0] = acc_1;
                    acc[(x - 1) * 2 + 1] = acc_2;
                } else {
                    acc[(x - 1) * 2 + 0] += acc_1;
                    acc[(x - 1) * 2 + 1] += acc_2;
                }
            }
        };
        inner(0);
        for (size_t needle_y = 1; needle_y < n_h; needle_y++) {
            inner(needle_y);
        }

        for (size_t x = 1; x < x_searches; x++) {
            uint32_t s_p = patch_sum[y * r_w + x];
            if (s_p == 0) {
                continue;
            }

            uint32_t acc_ = acc[(x - 1) * 2 + 0];
            int64_t num_i64 = (int64_t)n * (int64_t)acc_ - ((int64_t)s_n_1 * (int64_t)s_p);
            if (num_i64 > 0) {
                double num = (double)acc_ - (double)((uint64_t)s_n_1 * (uint64_t)s_p) / (double)n;
                double norm2_p = patch_norm[y * r_w + x];
                double den = sqrt(norm2_n_1 * norm2_p);
                double similarity = num / den;

                if (similarity > threshold) {
                    *out_cur++ = {(uint32_t)x, (uint32_t)y, (uint32_t)n_w, (uint32_t)n_h, (float)similarity};
                    n_written += 1;
                    if (out_cur == out_fin) {
                        return n_written;
                    }
                }
            }

            acc_ = acc[(x - 1) * 2 + 1];
            num_i64 = (int64_t)n * (int64_t)acc_ - ((int64_t)s_n_2 * (int64_t)s_p);
            if (num_i64 > 0) {
                double num = (double)acc_ - (double)((uint64_t)s_n_2 * (uint64_t)s_p) / (double)n;
                double norm2_p = patch_norm[y * r_w + x];
                double den = sqrt(norm2_n_2 * norm2_p);
                double similarity = num / den;

                if (similarity > threshold) {
                    *out_cur++ = {(uint32_t)x, (uint32_t)y, (uint32_t)n_w, (uint32_t)n_h, (float)similarity};
                    n_written += 1;
                    if (out_cur == out_fin) {
                        return n_written;
                    }
                }
            }
        }
    }

    return n_written;
}
