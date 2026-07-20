// Tests for KV cache compaction math utilities
//
// Validates the pure math functions used by the Attention Matching algorithm:
//   - Matrix multiplication variants (ABt, AtB)
//   - Softmax and stable exp
//   - Non-negative least squares (NNLS)
//   - Regularized least squares
//   - Full compaction pipeline (compact_head_highest_attn)

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "kv-compact-math.h"

static const float EPS = 1e-5f;

static bool approx_eq(float a, float b, float tol = EPS) {
    return fabsf(a - b) < tol;
}

// ============================================================================
// mat_mul_ABt tests
// ============================================================================

static void test_mat_mul_ABt_identity() {
    printf("  test_mat_mul_ABt_identity...");
    // A = [[1,0],[0,1]], B = [[1,0],[0,1]]
    // A * B^T = I * I^T = I
    float A[] = {1, 0, 0, 1};
    float B[] = {1, 0, 0, 1};
    float C[4] = {};
    mat_mul_ABt(A, B, C, 2, 2, 2);
    assert(approx_eq(C[0], 1.0f));
    assert(approx_eq(C[1], 0.0f));
    assert(approx_eq(C[2], 0.0f));
    assert(approx_eq(C[3], 1.0f));
    printf(" OK\n");
}

static void test_mat_mul_ABt_rectangular() {
    printf("  test_mat_mul_ABt_rectangular...");
    // A = [[1,2,3]], B = [[4,5,6],[7,8,9]]
    // A(1x3) * B^T(3x2) = (1x2)
    // C[0,0] = 1*4+2*5+3*6 = 32
    // C[0,1] = 1*7+2*8+3*9 = 50
    float A[] = {1, 2, 3};
    float B[] = {4, 5, 6, 7, 8, 9};
    float C[2] = {};
    mat_mul_ABt(A, B, C, 1, 2, 3);
    assert(approx_eq(C[0], 32.0f));
    assert(approx_eq(C[1], 50.0f));
    printf(" OK\n");
}

// ============================================================================
// mat_mul_AtB tests
// ============================================================================

static void test_mat_mul_AtB_basic() {
    printf("  test_mat_mul_AtB_basic...");
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // A^T * B where A is (2x2), B is (2x2), result (2x2)
    // A^T = [[1,3],[2,4]]
    // C[0,0] = 1*5+3*7 = 26, C[0,1] = 1*6+3*8 = 30
    // C[1,0] = 2*5+4*7 = 38, C[1,1] = 2*6+4*8 = 44
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4] = {};
    mat_mul_AtB(A, B, C, 2, 2, 2);
    assert(approx_eq(C[0], 26.0f));
    assert(approx_eq(C[1], 30.0f));
    assert(approx_eq(C[2], 38.0f));
    assert(approx_eq(C[3], 44.0f));
    printf(" OK\n");
}

static void test_mat_mul_AtB_rectangular() {
    printf("  test_mat_mul_AtB_rectangular...");
    // A is (3x2), B is (3x1) -> result is (2x1)
    // A = [[1,2],[3,4],[5,6]], B = [[1],[2],[3]]
    // A^T = [[1,3,5],[2,4,6]]
    // C[0] = 1*1+3*2+5*3 = 22
    // C[1] = 2*1+4*2+6*3 = 28
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {1, 2, 3};
    float C[2] = {};
    mat_mul_AtB(A, B, C, 3, 2, 1);
    assert(approx_eq(C[0], 22.0f));
    assert(approx_eq(C[1], 28.0f));
    printf(" OK\n");
}

// ============================================================================
// softmax_rows tests
// ============================================================================

static void test_softmax_rows_sums_to_one() {
    printf("  test_softmax_rows_sums_to_one...");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f, 0.0f, 2.0f};
    softmax_rows(data, 2, 4);

    // Each row should sum to 1
    float sum0 = data[0] + data[1] + data[2] + data[3];
    float sum1 = data[4] + data[5] + data[6] + data[7];
    assert(approx_eq(sum0, 1.0f));
    assert(approx_eq(sum1, 1.0f));
    printf(" OK\n");
}

static void test_softmax_rows_ordering() {
    printf("  test_softmax_rows_ordering...");
    float data[] = {1.0f, 2.0f, 3.0f};
    softmax_rows(data, 1, 3);

    // Larger input -> larger probability
    assert(data[0] < data[1]);
    assert(data[1] < data[2]);
    // All positive
    assert(data[0] > 0.0f);
    printf(" OK\n");
}

static void test_softmax_rows_uniform() {
    printf("  test_softmax_rows_uniform...");
    float data[] = {5.0f, 5.0f, 5.0f, 5.0f};
    softmax_rows(data, 1, 4);

    // Equal inputs -> uniform distribution
    for (int i = 0; i < 4; i++) {
        assert(approx_eq(data[i], 0.25f));
    }
    printf(" OK\n");
}

static void test_softmax_rows_numerical_stability() {
    printf("  test_softmax_rows_numerical_stability...");
    // Large values that would overflow without max-shift
    float data[] = {1000.0f, 1001.0f, 1002.0f};
    softmax_rows(data, 1, 3);

    float sum = data[0] + data[1] + data[2];
    assert(approx_eq(sum, 1.0f));
    assert(data[2] > data[1]);
    assert(data[1] > data[0]);
    printf(" OK\n");
}

// ============================================================================
// exp_rows_stable tests
// ============================================================================

static void test_exp_rows_stable_basic() {
    printf("  test_exp_rows_stable_basic...");
    float data[] = {0.0f, 1.0f, 2.0f};
    float row_sums[1];
    exp_rows_stable(data, row_sums, 1, 3);

    // After max-shift (max=2): exp(-2), exp(-1), exp(0) = 1
    assert(approx_eq(data[2], 1.0f));
    assert(data[1] < data[2]);
    assert(data[0] < data[1]);

    // row_sum should equal sum of the exp values
    float expected_sum = data[0] + data[1] + data[2];
    assert(approx_eq(row_sums[0], expected_sum));
    printf(" OK\n");
}

static void test_exp_rows_stable_large_values() {
    printf("  test_exp_rows_stable_large_values...");
    float data[] = {500.0f, 501.0f};
    float row_sums[1];
    exp_rows_stable(data, row_sums, 1, 2);

    // Should not overflow/NaN
    assert(std::isfinite(data[0]));
    assert(std::isfinite(data[1]));
    assert(std::isfinite(row_sums[0]));
    // Max element (501) after shift -> exp(0) = 1
    assert(approx_eq(data[1], 1.0f));
    printf(" OK\n");
}

// ============================================================================
// nnls_solve tests
// ============================================================================

static void test_nnls_identity() {
    printf("  test_nnls_identity...");
    // A = I (2x2), b = [3, 5]
    // min ||w - b||^2 s.t. w >= 0 => w = [3, 5]
    float A[] = {1, 0, 0, 1};
    float b[] = {3.0f, 5.0f};
    float w[2] = {};
    nnls_solve(A, b, w, 2, 2, 500);
    assert(approx_eq(w[0], 3.0f, 0.1f));
    assert(approx_eq(w[1], 5.0f, 0.1f));
    printf(" OK\n");
}

static void test_nnls_non_negative_constraint() {
    printf("  test_nnls_non_negative_constraint...");
    // A = I, b = [-2, 4]
    // Unconstrained solution = [-2, 4], but NNLS clamps to w >= 0
    // So w ≈ [0, 4]
    float A[] = {1, 0, 0, 1};
    float b[] = {-2.0f, 4.0f};
    float w[2] = {};
    nnls_solve(A, b, w, 2, 2, 500);
    assert(w[0] < 0.01f);  // should be ~0
    assert(approx_eq(w[1], 4.0f, 0.1f));
    printf(" OK\n");
}

static void test_nnls_overdetermined() {
    printf("  test_nnls_overdetermined...");
    // A = [[1],[1],[1]], b = [2, 3, 4]
    // LS solution: w = mean(b) = 3, and it's positive, so NNLS = 3
    float A[] = {1, 1, 1};
    float b[] = {2.0f, 3.0f, 4.0f};
    float w[1] = {};
    nnls_solve(A, b, w, 3, 1, 500);
    assert(approx_eq(w[0], 3.0f, 0.2f));
    printf(" OK\n");
}

// ============================================================================
// least_squares_solve tests
// ============================================================================

static void test_least_squares_identity() {
    printf("  test_least_squares_identity...");
    // A = I (2x2), b = [3, 7] -> x = [3, 7]
    float A[] = {1, 0, 0, 1};
    float b[] = {3.0f, 7.0f};
    float x[2] = {};
    least_squares_solve(A, b, x, 2, 2, 1, 0.0f);
    assert(approx_eq(x[0], 3.0f, 0.01f));
    assert(approx_eq(x[1], 7.0f, 0.01f));
    printf(" OK\n");
}

static void test_least_squares_overdetermined() {
    printf("  test_least_squares_overdetermined...");
    // A = [[1,0],[0,1],[1,1]], b = [[1],[0],[2]] (3x1 multi-column)
    // LS: A^T A = [[2,1],[1,2]], A^T b = [3, 2]
    // Solve: x = [4/3, 1/3]
    float A[] = {1, 0, 0, 1, 1, 1};
    float b[] = {1.0f, 0.0f, 2.0f};
    float x[2] = {};
    least_squares_solve(A, b, x, 3, 2, 1, 0.0f);
    assert(approx_eq(x[0], 4.0f / 3.0f, 0.01f));
    assert(approx_eq(x[1], 1.0f / 3.0f, 0.01f));
    printf(" OK\n");
}

static void test_least_squares_multi_rhs() {
    printf("  test_least_squares_multi_rhs...");
    // A = I (2x2), b = [[1,2],[3,4]] (2 rows, 2 columns) -> x = b
    float A[] = {1, 0, 0, 1};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[4] = {};
    least_squares_solve(A, b, x, 2, 2, 2, 0.0f);
    assert(approx_eq(x[0], 1.0f, 0.01f));
    assert(approx_eq(x[1], 2.0f, 0.01f));
    assert(approx_eq(x[2], 3.0f, 0.01f));
    assert(approx_eq(x[3], 4.0f, 0.01f));
    printf(" OK\n");
}

static void test_least_squares_with_ridge() {
    printf("  test_least_squares_with_ridge...");
    // With ridge > 0, solution should be shrunk toward zero
    float A[] = {1, 0, 0, 1};
    float b[] = {10.0f, 10.0f};
    float x_no_ridge[2] = {};
    float x_ridge[2] = {};
    least_squares_solve(A, b, x_no_ridge, 2, 2, 1, 0.0f);
    least_squares_solve(A, b, x_ridge, 2, 2, 1, 1.0f);
    // Ridge should shrink toward zero
    assert(fabsf(x_ridge[0]) < fabsf(x_no_ridge[0]));
    assert(fabsf(x_ridge[1]) < fabsf(x_no_ridge[1]));
    printf(" OK\n");
}

// ============================================================================
// compact_head_highest_attn tests
// ============================================================================

static void test_compact_no_compression_needed() {
    printf("  test_compact_no_compression_needed...");
    // t >= T means no compaction
    const int T = 4, n_q = 2, d_k = 3, d_v = 3;
    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < T * d_v; i++) V[i] = (float)(i % 5) * 0.2f;
    for (int i = 0; i < n_q * d_k; i++) Q[i] = (float)(i % 3) * 0.3f;

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, T);
    assert((int)result.selected_indices.size() == T);
    for (int i = 0; i < T; i++) {
        assert(result.selected_indices[i] == i);
    }
    // Beta should be all zeros
    for (int i = 0; i < T; i++) {
        assert(approx_eq(result.beta[i], 0.0f));
    }
    // C_v should equal V
    for (int i = 0; i < T * d_v; i++) {
        assert(approx_eq(result.C_v[i], V[i]));
    }
    printf(" OK\n");
}

static void test_compact_selects_correct_count() {
    printf("  test_compact_selects_correct_count...");
    const int T = 20, n_q = 8, d_k = 4, d_v = 4, t = 5;

    // Create keys with one obvious "hot" key that all queries attend to
    std::vector<float> K(T * d_k, 0.0f);
    std::vector<float> V(T * d_v, 0.0f);
    std::vector<float> Q(n_q * d_k, 0.0f);

    // Make diverse keys
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < d_k; d++) {
            K[i * d_k + d] = sinf((float)(i * d_k + d) * 0.7f);
        }
        for (int d = 0; d < d_v; d++) {
            V[i * d_v + d] = cosf((float)(i * d_v + d) * 0.3f);
        }
    }
    for (int i = 0; i < n_q; i++) {
        for (int d = 0; d < d_k; d++) {
            Q[i * d_k + d] = sinf((float)(i * d_k + d) * 1.1f);
        }
    }

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    // Check sizes
    assert((int)result.selected_indices.size() == t);
    assert((int)result.beta.size() == t);
    assert((int)result.C_v.size() == t * d_v);

    // Selected indices should be sorted and within range
    for (int i = 0; i < t; i++) {
        assert(result.selected_indices[i] >= 0);
        assert(result.selected_indices[i] < T);
    }
    for (int i = 1; i < t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }
    printf(" OK\n");
}

static void test_compact_quality_improves_with_refitting() {
    printf("  test_compact_quality_improves_with_refitting...");
    // Core claim: AM with value refitting should produce lower error than
    // simple token eviction (keeping original V for selected keys)

    const int T = 32, n_q = 16, d_k = 8, d_v = 8, t = 8;

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    // Create data with some structure
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < d_k; d++) {
            K[i * d_k + d] = sinf((float)(i * 3 + d) * 0.5f) + 0.1f * cosf((float)(i + d * 7));
        }
        for (int d = 0; d < d_v; d++) {
            V[i * d_v + d] = cosf((float)(i * 2 + d) * 0.3f) + 0.2f * sinf((float)(i + d * 5));
        }
    }
    for (int i = 0; i < n_q; i++) {
        for (int d = 0; d < d_k; d++) {
            Q[i * d_k + d] = sinf((float)(i * 5 + d) * 0.8f);
        }
    }

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    // Compute original attention output for a test query
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    std::vector<float> test_q(Q.data(), Q.data() + d_k); // first query

    // Original scores and softmax
    std::vector<float> orig_scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += test_q[d] * K[j * d_k + d];
        orig_scores[j] = dot * inv_sqrt_dk;
    }
    std::vector<float> orig_attn(orig_scores);
    softmax_rows(orig_attn.data(), 1, T);

    // Original output
    std::vector<float> orig_out(d_v, 0.0f);
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < d_v; d++) {
            orig_out[d] += orig_attn[j] * V[j * d_v + d];
        }
    }

    // Compacted scores with beta
    std::vector<float> comp_scores(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot += test_q[d] * K[result.selected_indices[j] * d_k + d];
        }
        comp_scores[j] = dot * inv_sqrt_dk + result.beta[j];
    }
    softmax_rows(comp_scores.data(), 1, t);

    // Output with refitted values (C_v)
    std::vector<float> refit_out(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) {
            refit_out[d] += comp_scores[j] * result.C_v[j * d_v + d];
        }
    }

    // Output without refitting (original V at selected indices)
    std::vector<float> evict_out(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) {
            evict_out[d] += comp_scores[j] * V[result.selected_indices[j] * d_v + d];
        }
    }

    // Compute MSE for both
    float mse_refit = 0.0f, mse_evict = 0.0f;
    for (int d = 0; d < d_v; d++) {
        float dr = refit_out[d] - orig_out[d];
        float de = evict_out[d] - orig_out[d];
        mse_refit += dr * dr;
        mse_evict += de * de;
    }
    mse_refit /= d_v;
    mse_evict /= d_v;

    printf("\n    MSE with refitting: %.8f\n", mse_refit);
    printf("    MSE without refit:  %.8f\n", mse_evict);
    printf("    Improvement:        %.2fx\n", mse_evict / (mse_refit + 1e-12f));

    // Value refitting should improve or at least not worsen the output
    // (in practice it should be significantly better for 4x compression)
    assert(mse_refit <= mse_evict * 1.1f);  // allow 10% tolerance for edge cases
    printf("  OK\n");
}

static void test_compact_beta_values_are_finite() {
    printf("  test_compact_beta_values_are_finite...");
    const int T = 16, n_q = 8, d_k = 4, d_v = 4, t = 4;

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = sinf((float)i * 0.3f);
    for (int i = 0; i < T * d_v; i++) V[i] = cosf((float)i * 0.2f);
    for (int i = 0; i < n_q * d_k; i++) Q[i] = sinf((float)i * 0.5f);

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    for (int j = 0; j < t; j++) {
        assert(std::isfinite(result.beta[j]));
    }
    for (int j = 0; j < t * d_v; j++) {
        assert(std::isfinite(result.C_v[j]));
    }
    printf(" OK\n");
}

static void test_compact_cosine_similarity() {
    printf("  test_compact_cosine_similarity...");
    // After compaction, the attention output should have high cosine similarity
    // with the original output

    const int T = 24, n_q = 12, d_k = 6, d_v = 6, t = 12; // 50% compression

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = sinf((float)i * 0.4f);
    for (int i = 0; i < T * d_v; i++) V[i] = cosf((float)i * 0.25f);
    for (int i = 0; i < n_q * d_k; i++) Q[i] = sinf((float)i * 0.6f);

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Test with the first reference query
    std::vector<float> test_q(Q.data(), Q.data() + d_k);

    // Original output
    std::vector<float> orig_scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += test_q[d] * K[j * d_k + d];
        orig_scores[j] = dot * inv_sqrt_dk;
    }
    softmax_rows(orig_scores.data(), 1, T);
    std::vector<float> orig_out(d_v, 0.0f);
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < d_v; d++) orig_out[d] += orig_scores[j] * V[j * d_v + d];
    }

    // Compacted output
    std::vector<float> comp_scores(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot += test_q[d] * K[result.selected_indices[j] * d_k + d];
        }
        comp_scores[j] = dot * inv_sqrt_dk + result.beta[j];
    }
    softmax_rows(comp_scores.data(), 1, t);
    std::vector<float> comp_out(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) comp_out[d] += comp_scores[j] * result.C_v[j * d_v + d];
    }

    // Cosine similarity
    float dot_prod = 0.0f, norm_orig = 0.0f, norm_comp = 0.0f;
    for (int d = 0; d < d_v; d++) {
        dot_prod += orig_out[d] * comp_out[d];
        norm_orig += orig_out[d] * orig_out[d];
        norm_comp += comp_out[d] * comp_out[d];
    }
    float cos_sim = dot_prod / (sqrtf(norm_orig * norm_comp) + 1e-8f);

    printf("\n    Cosine similarity (50%% compression): %.6f\n", cos_sim);
    // At 50% compression, cosine similarity should be very high
    assert(cos_sim > 0.9f);
    printf("  OK\n");
}

// ============================================================================
// compact_layer_all_heads tests
// ============================================================================

static void test_compact_layer_shared_selection() {
    printf("  test_compact_layer_shared_selection...");
    // Multiple heads should share the same selected indices
    const int T = 16, n_q = 8, n_head_kv = 2, d_k = 4, d_v = 4, t = 4;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = sinf((float)i * 0.3f);
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = cosf((float)i * 0.2f);
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = sinf((float)i * 0.5f);

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    // Check shared selection
    assert((int)result.selected_indices.size() == t);
    for (int i = 0; i < t; i++) {
        assert(result.selected_indices[i] >= 0);
        assert(result.selected_indices[i] < T);
    }
    // Sorted
    for (int i = 1; i < t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }

    // Check per-head outputs exist with correct sizes
    assert((int)result.beta.size() == n_head_kv);
    assert((int)result.C_v.size() == n_head_kv);
    for (int h = 0; h < n_head_kv; h++) {
        assert((int)result.beta[h].size() == t);
        assert((int)result.C_v[h].size() == t * d_v);
    }

    printf(" OK\n");
}

static void test_compact_layer_no_compression() {
    printf("  test_compact_layer_no_compression...");
    const int T = 4, n_q = 2, n_head_kv = 2, d_k = 3, d_v = 3;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = (float)(i % 5) * 0.2f;
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = (float)(i % 3) * 0.3f;

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, T);

    // t >= T → no compaction, all indices selected
    assert((int)result.selected_indices.size() == T);
    for (int i = 0; i < T; i++) assert(result.selected_indices[i] == i);

    // Beta should be all zeros
    for (int h = 0; h < n_head_kv; h++) {
        for (int i = 0; i < T; i++) {
            assert(approx_eq(result.beta[h][i], 0.0f));
        }
    }

    printf(" OK\n");
}

static void test_compact_layer_quality_per_head() {
    printf("  test_compact_layer_quality_per_head...");
    // Verify each head's output quality improves with refitting
    const int T = 32, n_q = 16, n_head_kv = 3, d_k = 6, d_v = 6, t = 8;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = sinf((float)(i * 3 + 1) * 0.4f);
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = cosf((float)(i * 2 + 3) * 0.3f);
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = sinf((float)(i * 5 + 2) * 0.7f);

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    for (int h = 0; h < n_head_kv; h++) {
        // Test query: first Q for this head
        const float * q_test = Q.data() + h * d_k;

        // Original output
        std::vector<float> orig_scores(T);
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += q_test[d] * K[j * n_embd_k_gqa + h * d_k + d];
            }
            orig_scores[j] = dot * inv_sqrt_dk;
        }
        softmax_rows(orig_scores.data(), 1, T);

        std::vector<float> orig_out(d_v, 0.0f);
        for (int j = 0; j < T; j++) {
            for (int d = 0; d < d_v; d++) {
                orig_out[d] += orig_scores[j] * V[j * n_embd_v_gqa + h * d_v + d];
            }
        }

        // Compacted output with beta + C_v
        std::vector<float> comp_scores(t);
        for (int j = 0; j < t; j++) {
            float dot = 0.0f;
            int idx = result.selected_indices[j];
            for (int d = 0; d < d_k; d++) {
                dot += q_test[d] * K[idx * n_embd_k_gqa + h * d_k + d];
            }
            comp_scores[j] = dot * inv_sqrt_dk + result.beta[h][j];
        }
        softmax_rows(comp_scores.data(), 1, t);

        std::vector<float> comp_out(d_v, 0.0f);
        for (int j = 0; j < t; j++) {
            for (int d = 0; d < d_v; d++) {
                comp_out[d] += comp_scores[j] * result.C_v[h][j * d_v + d];
            }
        }

        // Cosine similarity should be decent (> 0.8 at 4x compression)
        float dot_p = 0.0f, n_o = 0.0f, n_c = 0.0f;
        for (int d = 0; d < d_v; d++) {
            dot_p += orig_out[d] * comp_out[d];
            n_o += orig_out[d] * orig_out[d];
            n_c += comp_out[d] * comp_out[d];
        }
        float cos_sim = dot_p / (sqrtf(n_o * n_c) + 1e-8f);

        printf("\n    Head %d: cos_sim=%.6f", h, cos_sim);
        assert(cos_sim > 0.8f);
    }

    printf("\n  OK\n");
}

static void test_compact_layer_finite_values() {
    printf("  test_compact_layer_finite_values...");
    const int T = 16, n_q = 8, n_head_kv = 4, d_k = 4, d_v = 4, t = 4;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = sinf((float)i * 0.1f);
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = cosf((float)i * 0.15f);
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = sinf((float)i * 0.25f);

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < t; j++) {
            assert(std::isfinite(result.beta[h][j]));
        }
        for (int j = 0; j < t * d_v; j++) {
            assert(std::isfinite(result.C_v[h][j]));
        }
    }

    printf(" OK\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("test-kv-compact-math:\n");

    printf("\n=== Matrix multiplication ===\n");
    test_mat_mul_ABt_identity();
    test_mat_mul_ABt_rectangular();
    test_mat_mul_AtB_basic();
    test_mat_mul_AtB_rectangular();

    printf("\n=== Softmax ===\n");
    test_softmax_rows_sums_to_one();
    test_softmax_rows_ordering();
    test_softmax_rows_uniform();
    test_softmax_rows_numerical_stability();

    printf("\n=== Stable exp ===\n");
    test_exp_rows_stable_basic();
    test_exp_rows_stable_large_values();

    printf("\n=== NNLS solver ===\n");
    test_nnls_identity();
    test_nnls_non_negative_constraint();
    test_nnls_overdetermined();

    printf("\n=== Least squares solver ===\n");
    test_least_squares_identity();
    test_least_squares_overdetermined();
    test_least_squares_multi_rhs();
    test_least_squares_with_ridge();

    printf("\n=== Full compaction pipeline ===\n");
    test_compact_no_compression_needed();
    test_compact_selects_correct_count();
    test_compact_quality_improves_with_refitting();
    test_compact_beta_values_are_finite();
    test_compact_cosine_similarity();

    printf("\n=== Layer-level compaction ===\n");
    test_compact_layer_shared_selection();
    test_compact_layer_no_compression();
    test_compact_layer_quality_per_head();
    test_compact_layer_finite_values();

    printf("\nAll tests passed!\n");
    return 0;
}
