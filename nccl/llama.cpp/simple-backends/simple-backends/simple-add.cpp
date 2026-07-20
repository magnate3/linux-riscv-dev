#include "ggml.h"
#include "ggml-cpu.h"
#include <cstring>
#include <iostream>
#include <vector>

int main () {
    struct ggml_init_params params {
        /*.mem_size   =*/ 1024 * 1024 * 1024 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);

    // --------------------
    // Modify this section to test different tensor computations
    float a_data[3 * 2] = {
        1, 2,
        3, 4,
        5, 6
    };

    float b_data[3 * 2] = {
        1, 1,
        1, 1,
        1, 1
    };


    // 因为ggml中tensor表示和pytorch相反，因此3行2列的矩阵，表示为[2, 3]
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    memcpy(a->data, a_data, ggml_nbytes(a));
    memcpy(b->data, b_data, ggml_nbytes(b));

    ggml_tensor* result = ggml_add(ctx, a, b);
    // --------------------

    struct ggml_cgraph  * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    ggml_graph_compute_with_ctx(ctx, gf, 1);

    std::vector<float> out_data(ggml_nelements(result));
    memcpy(out_data.data(), result->data, ggml_nbytes(result));

    ggml_free(ctx);
    return 0;
}
