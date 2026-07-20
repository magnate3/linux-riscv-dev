#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

#ifdef GGML_USE_VULKAN
#include <ggml-vulkan.h>
#endif

#include <cmath>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct magnet_hparams {
    int32_t dim = 1024;
    int32_t num_heads = 16;
    int32_t num_layers = 24;
    int32_t hidden_scale = 4;
    int32_t n_q = 4;
    int32_t kv_repeat = 1;
    int32_t card = 2048;
    int32_t subcodes_context = 5;
    int32_t sample_rate = 32000;
};

struct magnet_transformer_block {
    // Self MHA
    // The q, k, v matricies are derived from this tensor
    struct ggml_tensor* self_attn_in_proj_w;
    // nn.Linear applied to output of attention
    struct ggml_tensor* self_attn_out_proj_w;

    // nn.Linear
    struct ggml_tensor* linear1_w;
    struct ggml_tensor* linear2_w;

    // nn.LayerNorm
    // elementwise_affine=True
    struct ggml_tensor* layer_norm1_w;
    struct ggml_tensor* layer_norm1_b;

    // nn.LayerNorm
    // elementwise_affine=True
    struct ggml_tensor* layer_norm2_w;
    struct ggml_tensor* layer_norm2_b;

    // Cross MHA
    struct ggml_tensor* cross_attn_in_proj_w;
    struct ggml_tensor* cross_attn_out_proj_w;

    // nn.LayerNorm
    // elementwise_affine=True
    struct ggml_tensor* norm_cross_w;
    struct ggml_tensor* norm_cross_b;
};

struct magnet_transformer {
    std::vector<magnet_transformer_block> transformer_blocks;
};

struct magnet_model {
    // See audiocraft T5Conditioner
    struct ggml_tensor* conditioning_w;
    struct ggml_tensor* conditioning_b;

    // nn.Embedding
    // Scaled embedding for n_q codebooks
    struct ggml_tensor* embed0_w;
    struct ggml_tensor* embed1_w;
    struct ggml_tensor* embed2_w;
    struct ggml_tensor* embed3_w;

    magnet_transformer transformer;

    // nn.LayerNorm
    struct ggml_tensor* out_norm_w;
    struct ggml_tensor* out_norm_b;

    // nn.Linear w/o bias for n_q codebooks
    struct ggml_tensor* linear0_w;
    struct ggml_tensor* linear1_w;
    struct ggml_tensor* linear2_w;
    struct ggml_tensor* linear3_w;

    magnet_hparams hparams;
    struct ggml_context* ctx;
    ggml_backend_t backend;
    struct ggml_backend_buffer* buffer;
};

struct magnet_context {
    magnet_model model;

    struct ggml_gallocr* galloc;
    ggml_tallocr talloc;
};

static void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data)
{
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

#define MAGNET_INFILE_MAGIC 0x46554747 // 'GGUF' LE
#define GGUF_GET_I32(ctx, key) gguf_get_val_i32(ctx, gguf_find_key(ctx, key))

// FIXME: remove this
#define MAX_PRINT_ELEMENTS 10

void print_tensor(struct ggml_tensor* tensor)
{
    if (tensor == nullptr) {
        printf("tensor is null\n");
        return;
    }

    int n_dims = ggml_n_dims(tensor);
    int64_t* ne = tensor->ne;

    printf("\ntensor(");
    for (int i = 0; i < n_dims; i++) {
        printf("%lld%s", ne[i], i < n_dims - 1 ? ", " : "");
    }
    printf(", type=%s) = ", ggml_type_name(tensor->type));

    if (ggml_nelements(tensor) == 0) {
        printf("[]\n");
        return;
    }

    int max_per_dim = MAX_PRINT_ELEMENTS;
    int* indices = (int*)calloc(n_dims, sizeof(int));
    bool* first_in_dim = (bool*)calloc(n_dims, sizeof(bool));
    for (int i = 0; i < n_dims; i++) {
        first_in_dim[i] = true;
    }
    int depth = 0;

    while (depth >= 0) {
        if (depth == n_dims) {
            int64_t idx = 0;
            int64_t mult = 1;
            for (int i = 0; i < n_dims; i++) {
                idx += indices[i] * mult;
                mult *= ne[i];
            }
            if (!first_in_dim[depth - 1]) {
                printf(", ");
            }
            printf("%.4f", ggml_get_f32_1d(tensor, idx));
            first_in_dim[depth - 1] = false;

            depth--;
            indices[depth]++;
        } else if (indices[depth] < ne[depth] && (indices[depth] < max_per_dim || indices[depth] >= ne[depth] - max_per_dim)) {
            if (indices[depth] == 0) {
                if (!first_in_dim[depth]) {
                    printf(",\n");
                    for (int i = 0; i < depth; i++)
                        printf(" ");
                }
                printf("[");
                first_in_dim[depth] = true;
            }
            depth++;
        } else {
            if (indices[depth] > 0) {
                if (ne[depth] > 2 * max_per_dim) {
                    printf(", ...");
                }
                printf("]");
            }
            first_in_dim[depth] = false;
            indices[depth] = 0;
            depth--;
            if (depth >= 0)
                indices[depth]++;
        }
    }
    printf("\n");

    free(indices);
    free(first_in_dim);
}

bool load_parameters(std::string& file_name, magnet_model& model)
{
    // Load le model
    {
        struct ggml_init_params params = {
            .mem_size = 0,
            .mem_buffer = NULL,
        };
        model.ctx = ggml_init(params);

#ifdef GGML_USE_VULKAN
        ggml_vk_instance_init()
            model.backend
            = ggml_backend_vk_init(0);
#endif

        if (!model.backend) {
            model.backend = ggml_backend_cpu_init();
        }

        // Now try to init from the file
        struct gguf_init_params gguf_params {
            .no_alloc = false,
            .ctx = &model.ctx,
        };

        struct gguf_context* gguf_ctx = gguf_init_from_file(file_name.c_str(), gguf_params);
        if (gguf_ctx == nullptr) {
            fprintf(stderr, "%s: Failed to load gguf file\n", __func__);
            return false;
        }

        model.hparams.dim = GGUF_GET_I32(gguf_ctx, "params.dim");
        model.hparams.num_heads = GGUF_GET_I32(gguf_ctx, "params.num_heads");
        model.hparams.num_layers = GGUF_GET_I32(gguf_ctx, "params.num_layers");
        model.hparams.hidden_scale = GGUF_GET_I32(gguf_ctx, "params.hidden_scale");
        model.hparams.n_q = GGUF_GET_I32(gguf_ctx, "params.n_q");
        model.hparams.kv_repeat = GGUF_GET_I32(gguf_ctx, "params.kv_repeat");
        model.hparams.card = GGUF_GET_I32(gguf_ctx, "params.card");
        model.hparams.subcodes_context = GGUF_GET_I32(gguf_ctx, "params.subcodes_context");
        model.hparams.sample_rate = GGUF_GET_I32(gguf_ctx, "params.sample_rate");

        printf("Model Hyperparameters\n");
        printf("dim:                %d\n", model.hparams.dim);
        printf("num_heads:          %d\n", model.hparams.num_heads);
        printf("num_layers:         %d\n", model.hparams.num_layers);
        printf("n_q:                %d\n", model.hparams.n_q);
        printf("card:               %d\n", model.hparams.card);
        printf("hidden_scale:       %d\n", model.hparams.hidden_scale);
        printf("kv_repeat:          %d\n", model.hparams.kv_repeat);
        printf("subcodes_context:   %d\n", model.hparams.subcodes_context);
        printf("sample_rate:        %d\n", model.hparams.sample_rate);

        int n_keys = gguf_get_n_kv(gguf_ctx);
        printf("Number of keys: %d\n", n_keys);
        int n_tensors = gguf_get_n_tensors(gguf_ctx);
        printf("Number of tensors: %d\n", n_tensors);

        gguf_free(gguf_ctx);
    }

    {
        auto& hparams = model.hparams;
        ggml_context* ctx = model.ctx;
        auto n_q = hparams.n_q;

        GGML_ASSERT(ggml_get_first_tensor(ctx) != nullptr);

        // Embeddings
        // NOTE: can support more than n_q codebooks, however both small & medium models only use 4 so hardcoded
        model.embed0_w = ggml_get_tensor(ctx, "emb.0.weight");
        model.embed1_w = ggml_get_tensor(ctx, "emb.1.weight");
        model.embed2_w = ggml_get_tensor(ctx, "emb.2.weight");
        model.embed3_w = ggml_get_tensor(ctx, "emb.3.weight");
        printf("Embedding weight shape (%d, %d)\n", model.embed0_w->ne[0], model.embed0_w->ne[1]);

        // Linear Layers
        model.linear0_w = ggml_get_tensor(ctx, "linears.0.weight");
        model.linear1_w = ggml_get_tensor(ctx, "linears.1.weight");
        model.linear2_w = ggml_get_tensor(ctx, "linears.2.weight");
        model.linear3_w = ggml_get_tensor(ctx, "linears.3.weight");
        printf("Linear weight shape (%d, %d)\n", model.linear0_w->ne[0], model.linear0_w->ne[1]);

        // Normalization
        model.out_norm_w = ggml_get_tensor(ctx, "out_norm.weight");
        model.out_norm_b = ggml_get_tensor(ctx, "out_norm.bias");
        printf("out_norm weight & bias shape (%d) (%d)\n", model.out_norm_w->ne[0], model.out_norm_b->ne[0]);

        model.transformer = magnet_transformer();
        auto& transformer = model.transformer;

        // Reserve num_layers transformer blocks
        transformer.transformer_blocks.resize(hparams.num_layers);
        for (int i = 0; i < hparams.num_layers; i++) {
            auto& layer = transformer.transformer_blocks[i];
            char tmp_name[255];

#define CHECK_SHAPE(tensor) \
    GGML_ASSERT(tmp_name);  \
    // printf("%s shape: (%d, %d, %d, %d)\n", tmp_name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

            // Under the assumption that the layers are contiguous, save some time from lookup
            snprintf(tmp_name, 255, "transformer.layers.%d.self_attn.in_proj_weight", i);
            layer.self_attn_in_proj_w = ggml_get_tensor(ctx, tmp_name);
            CHECK_SHAPE(layer.self_attn_in_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.self_attn.out_proj.weight", i);
            layer.self_attn_out_proj_w = ggml_get_next_tensor(ctx, layer.self_attn_in_proj_w);
            CHECK_SHAPE(layer.self_attn_out_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.linear1", i);
            layer.linear1_w = ggml_get_next_tensor(ctx, layer.self_attn_out_proj_w);
            CHECK_SHAPE(layer.linear1_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.linear2", i);
            layer.linear2_w = ggml_get_next_tensor(ctx, layer.linear1_w);
            CHECK_SHAPE(layer.linear2_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm1.weight", i);
            layer.layer_norm1_w = ggml_get_next_tensor(ctx, layer.linear2_w);
            CHECK_SHAPE(layer.layer_norm1_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm1.bias", i);
            layer.layer_norm1_b = ggml_get_next_tensor(ctx, layer.layer_norm1_w);
            CHECK_SHAPE(layer.layer_norm1_b);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm2.weight", i);
            layer.layer_norm2_w = ggml_get_next_tensor(ctx, layer.layer_norm1_b);
            CHECK_SHAPE(layer.layer_norm2_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm2.bias", i);
            layer.layer_norm2_b = ggml_get_next_tensor(ctx, layer.layer_norm2_w);
            CHECK_SHAPE(layer.layer_norm2_b);

            snprintf(tmp_name, 255, "transformer.layers.%d.cross_attention.in_proj_weight", i);
            layer.cross_attn_in_proj_w = ggml_get_next_tensor(ctx, layer.layer_norm2_b);
            CHECK_SHAPE(layer.cross_attn_in_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.cross_attention.out_proj.weight", i);
            layer.cross_attn_out_proj_w = ggml_get_next_tensor(ctx, layer.cross_attn_in_proj_w);
            CHECK_SHAPE(layer.cross_attn_out_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm_cross.weight", i);
            layer.norm_cross_w = ggml_get_next_tensor(ctx, layer.cross_attn_out_proj_w);
            CHECK_SHAPE(layer.norm_cross_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm_cross.bias", i);
            layer.norm_cross_b = ggml_get_next_tensor(ctx, layer.norm_cross_w);
            CHECK_SHAPE(layer.norm_cross_b);
        }
    }

    return true;
}

#define PRINT_SHAPE(comment, tensor) printf("%s: (%d, %d, %d, %d)\n", comment, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

// 2.1) Normalize (LayerNorm https://arxiv.org/pdf/1607.06450)
// Input shape should be of (*, N) where N is the shape of the weight, output is (*, N) (same shape)
ggml_tensor* magnet_layer_norm_forward(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x)
{
    // layer_norm = ((x - mean) / sqrt(variance(x))) * weight + bias
    // The ggml_norm operation computes the norm without applying weight & bias
    // Reshape the weight and bias to apply over the last dimension
    GGML_ASSERT(ggml_n_dims(x) == 2); // for current use case only work on 2 dimensional tensors
    ggml_tensor* reshaped_w = ggml_reshape_2d(ctx, w, 1, w->ne[0]);
    ggml_tensor* reshaped_b = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5), reshaped_w), reshaped_b);
}

// Linear transformation layer
// Input shape: (*, Hin) Output shape: (*, Hout) where Hin is input features, Hout is output features
ggml_tensor* magnet_linear_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b = nullptr)
{
    ggml_tensor* out = ggml_mul_mat(ctx, w, ggml_cont(ctx, ggml_transpose(ctx, x)));
    out = ggml_cont(ctx, ggml_transpose(ctx, out));
    if (b != nullptr) {
        out = ggml_add(ctx, out, b);
    }
    return out;
}

// Scaled Dot-Product Attention
// Attention(Q, K, V) = softmax( (Q * K^T)/sqrt(dk) ) * V
ggml_tensor* magnet_dp_attn_forward(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v)
{
    GGML_ASSERT(k->ne[1] == 1024); // should be of shape (seq_len, embed_dim)
    auto scale = 1.0f / sqrtf(k->ne[1]);
    struct ggml_tensor* x = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, q)), ggml_cont(ctx, ggml_transpose(ctx, k))); // Q*K^T. Tech debt from loading transpoed weights
    x = ggml_scale(ctx, x, scale);
    x = ggml_soft_max(ctx, x);
    x = ggml_mul_mat(ctx, x, v);
    return x;
}

ggml_tensor* magnet_transformer_block_forward(magnet_model* model, ggml_context* ctx, magnet_transformer_block* block, ggml_tensor* x, ggml_tensor* cross_attn_src)
{
    const auto& hparams = model->hparams;
    auto embed_dim = hparams.dim;
    auto n_heads = hparams.num_heads;
    auto head_dim = embed_dim / n_heads;
    auto n_kv = hparams.kv_repeat;
    auto n_head_kv = n_heads / n_kv;
    auto n_batch = 1;

    // can cheat and just use the last dim
    // shape is B, T, C. normalize along dimension 1 (T) (in reference implementation)
    // Don't care about B, we doing inference in this bitch! Take your ass back to training!

    // Make the correct shape for the layer norm (only care about normalizing the T layers)
    // 2.1) Normalize (LayerNorm https://arxiv.org/pdf/1607.06450)
    {
        if (ggml_backend_is_cpu(model->backend)) {
            block->layer_norm1_w = ggml_cast(ctx, block->layer_norm1_w, GGML_TYPE_F32);
            block->layer_norm1_b = ggml_cast(ctx, block->layer_norm1_b, GGML_TYPE_F32);
        }

        x = magnet_layer_norm_forward(ctx, block->layer_norm1_w, block->layer_norm1_b, x);
    }

    // 2.2) Self attention (use Flash Attention, see paper https://arxiv.org/abs/2205.14135)
    {
        struct ggml_tensor* projected = magnet_linear_forward(ctx, x, block->self_attn_in_proj_w);

        // Unstack K, Q, V from projected
        auto seq_len = projected->ne[0];
        struct ggml_tensor* q = ggml_view_2d(ctx, projected, seq_len, embed_dim, projected->nb[1], 0);
        struct ggml_tensor* k = ggml_view_2d(ctx, projected, seq_len, embed_dim, projected->nb[1], projected->nb[1] * embed_dim); // projected[:, :embed_dim]
        struct ggml_tensor* v = ggml_view_2d(ctx, projected, seq_len, embed_dim, projected->nb[1], projected->nb[1] * 2 * embed_dim); // projected[:, :embed_dim]

        // FIXME: either implement flash attention from scratch or fix this operation
#ifdef FLASH_ATTN
        q = ggml_reshape_3d(ctx, q, head_dim, seq_len, n_heads);
        k = ggml_reshape_3d(ctx, k, head_dim, seq_len, n_head_kv);
        v = ggml_reshape_3d(ctx, v, head_dim, seq_len, n_head_kv);

        // NOTE: in order to use self_attn on CPU, must be of type F16 due to bug in GGML, no to_float function to convert k,v tensors
        if (ggml_backend_is_cpu(model->backend)) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }

        // FIXME: use mask provided by model input (get this from a magnet_context?)
        struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, GGML_PAD(n_batch, GGML_KQ_MASK_PAD));

        struct ggml_tensor* self_attn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.f / sqrtf(embed_dim), 0); // small: [64, 16, 1, 1]

        // then apply the out_proj
        self_attn = ggml_reshape_2d(ctx, self_attn, seq_len, embed_dim);
#else
        struct ggml_tensor* self_attn = magnet_dp_attn_forward(ctx, q, k, v);
#endif
        self_attn = magnet_linear_forward(ctx, self_attn, block->self_attn_out_proj_w);

        x = ggml_add(ctx, x, self_attn);
    }

    // 2.3) Cross attn normalization (LayerNorm). This is done with the provided conditions
    {
        if (ggml_backend_is_cpu(model->backend)) {
            block->norm_cross_w = ggml_cast(ctx, block->norm_cross_w, GGML_TYPE_F32);
            block->norm_cross_b = ggml_cast(ctx, block->norm_cross_b, GGML_TYPE_F32);
        }
        x = magnet_layer_norm_forward(ctx, block->norm_cross_w, block->norm_cross_b, x);
    }

    // 2.4) Cross attention
    {
        auto seq_len = x->ne[0];

        GGML_ASSERT(cross_attn_src->ne[0] == seq_len);
        GGML_ASSERT(cross_attn_src->ne[1] == embed_dim);

        // Audiocraft (reference MAGNeT implementation) has the query from input (x), keys & values from the conditioning tensor (cross_attn_src)
        // The weight tensor is stacked vectors, creating the shape [3 * embed_dim, embed_dim], slice to get weights for qkv (calc independently)
        struct ggml_tensor* q_slice = ggml_view_2d(ctx, block->cross_attn_in_proj_w, embed_dim, embed_dim, block->cross_attn_in_proj_w->nb[1], 0);
        struct ggml_tensor* q = magnet_linear_forward(ctx, x, q_slice);

        struct ggml_tensor* k_slice = ggml_view_1d(ctx, block->cross_attn_in_proj_w, embed_dim * embed_dim, embed_dim * embed_dim * ggml_element_size(block->cross_attn_in_proj_w));
        k_slice = ggml_reshape_2d(ctx, k_slice, embed_dim, embed_dim);
        struct ggml_tensor* k = magnet_linear_forward(ctx, cross_attn_src, k_slice);

        struct ggml_tensor* v_slice = ggml_view_1d(ctx, block->cross_attn_in_proj_w, embed_dim * embed_dim, 2 * embed_dim * embed_dim * ggml_element_size(block->cross_attn_in_proj_w));
        v_slice = ggml_reshape_2d(ctx, v_slice, embed_dim, embed_dim);
        struct ggml_tensor* v = magnet_linear_forward(ctx, cross_attn_src, v_slice);

#ifdef FLASH_ATTN
        q = ggml_reshape_3d(ctx, q, head_dim, seq_len, n_heads);
        k = ggml_reshape_3d(ctx, k, head_dim, seq_len, n_head_kv);
        v = ggml_reshape_3d(ctx, v, head_dim, seq_len, n_head_kv);

        // Hack to prevent segfaulting on CPU backend
        if (ggml_backend_is_cpu(model->backend)) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }

        // Apply the mask and attention op in the same manner
        // FIXME: use mask provided by model input (get this from a magnet_context?)
        struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, block->cross_attn_in_proj_w->type, n_kv, GGML_PAD(n_batch, GGML_KQ_MASK_PAD));

        struct ggml_tensor* cross_attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.f / sqrt(embed_dim), 0); // small: [64, 16, 1, 1]

        // then apply the out_proj
        cross_attn = ggml_reshape_2d(ctx, cross_attn, seq_len, embed_dim);
#else
        struct ggml_tensor* cross_attn = magnet_dp_attn_forward(ctx, q, k, v);
#endif
        cross_attn = magnet_linear_forward(ctx, cross_attn, block->cross_attn_out_proj_w);

        x = ggml_add(ctx, x, cross_attn);
    }

    // 2.5) Normalize (LayerNorm)
    {
        if (ggml_backend_is_cpu(model->backend)) {
            block->layer_norm2_w = ggml_cast(ctx, block->layer_norm2_w, GGML_TYPE_F32);
            block->layer_norm2_b = ggml_cast(ctx, block->layer_norm2_b, GGML_TYPE_F32);
        }

        x = magnet_layer_norm_forward(ctx, block->layer_norm2_w, block->layer_norm2_b, x);
    }

    // 2.6) Feedforward block (linears)
    {
        struct ggml_tensor* x_p = magnet_linear_forward(ctx, x, block->linear1_w);
        x_p = ggml_gelu(ctx, x_p);
        x_p = magnet_linear_forward(ctx, x_p, block->linear2_w);

        x = ggml_add(ctx, x, x_p);
    }

    return x;
}

// Positional encoding must be a custom operation
void magnet_positional_encoding(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata)
{
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    // at the moment only have 2 dimensions so don't care about other channels!
    const auto MAX_PERIOD = 10000;
    for (int i = 0; i < src->ne[0]; i++) {
        for (int pos = 0; pos < src->ne[i]; pos++) {
            float inner = pos / pow(MAX_PERIOD, (2.0 * i) / 1024.0);
            float val = (i % 2 == 0) ? cos(inner) : sin(inner);
            ggml_set_f32_nd(dst, i, pos, 0, 0, val);
        }
    }
}

// FIXME: remove this 
void magnet_test_fill(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata)
{
    auto B = src->ne[0];
    auto K = src->ne[1];
    auto S = src->ne[2];
    auto E = src->ne[3];

    for(int b = 0;b<B;b++) {
        for(int k = 0;k<K;k++) {
            for(int s = 0;s<S;s++) {
                for(int e = 0;e<E;e++) {
                    *(float*)((char*)dst->data + (b * dst->nb[0] + k * dst->nb[1] + s * dst->nb[2] + e * dst->nb[3])) = (float)(k + 1);
                }
            }
        }
    }
}

void magnet_embedding(struct ggml_tensor* dst, const struct ggml_tensor* src, const struct ggml_tensor* emb, int ith, int nth, void* userdata)
{
    GGML_ASSERT(emb->type == GGML_TYPE_F32);

    auto B = src->ne[0];
    auto K = src->ne[1];
    auto S = src->ne[2];

    for(int b = 0;b<B;b++) {
        for(int k=0;k<K;k++) {
            for(int s=0;s<S;s++) {
                // get the token id for particular token in sequence (this is the index of the embedding vector)
                int token_id = (int)*(float*)((char*)src->data + (s * src->nb[2] + k * src->nb[1] + b * src->nb[0]));
                // get the pointer to the position in the embedding weights where the vector for the index starts
                float* src_embedding = (float*)((char*)emb->data + token_id * emb->nb[1]);
                // get the pointer to the position in dst tensor where the emb vector will start
                float* dst_embedding = (float*)((char*)dst->data + (s * dst->nb[2] + k * dst->nb[1] + b * dst->nb[0]));

                // now copy the embedding vector to its corresponding position in the dst tensor
                for(int e = 0;e<emb->ne[0];e++) {
                    *(float*)((char*)dst_embedding + e * dst->nb[3]) = *(float*)((char*)src_embedding + e * emb->nb[0]);
                }
            }
        }
    }
}

ggml_tensor* magnet_transformer_forward(magnet_model* model, ggml_context* ctx, ggml_tallocr* alloc, ggml_tensor* x, ggml_tensor* cross_attn_src)
{
    // Expected shape (B, T, C)
    // FIXME: remove the batch_size dimension
    auto batch_size = x->ne[0];
    auto tokens = x->ne[1];
    auto channels = x->ne[2];

    // 1) Create position encodings (MAGNeT uses sine encodings)
    // FIXME: remove batch size
    ggml_tensor* positions = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, batch_size, tokens, channels);
    ggml_tallocr_alloc(alloc, positions);

    // Positional encoding!!!! :D
    positions = ggml_map_custom1(ctx, positions, magnet_positional_encoding, GGML_N_TASKS_MAX, NULL);
    PRINT_SHAPE("positions shape: ", positions);
    x = ggml_add(ctx, positions, x);
    PRINT_SHAPE("x shape before blocks: ", x);

    // 2) Apply each transformer Layer
    // FIXME: cross_attn_src should come from the conditioners

    auto& blocks = model->transformer.transformer_blocks;
    for (int i = 0; i < blocks.size(); i++) {
        GGML_ASSERT(blocks[i].linear1_w);
        x = magnet_transformer_block_forward(model, ctx, &blocks[i], x, cross_attn_src);
    }

    return x;
}

ggml_tensor* magnet_forward(magnet_model* model, ggml_context* ctx, ggml_tallocr* alloc, ggml_tensor* x)
{
    // Expected shape (B, K, S)
    auto batch_size = x->ne[0];
    auto codebooks = x->ne[1]; // Redundant since n_q is a hparam
    auto seq_len = x->ne[2];

    // Map tokens to embedding vectors for each codebook (4 by default)
    GGML_ASSERT(model->hparams.n_q == 4 && model->hparams.n_q == codebooks);

    // FIXME: check if this slicing is correct
    struct ggml_tensor* p = ggml_view_4d(ctx, x, batch_size, 1, seq_len, model->hparams.dim, x->nb[1], x->nb[2], x->nb[3], 0);
    struct ggml_tensor* q = ggml_view_4d(ctx, x, batch_size, 1, seq_len, model->hparams.dim, x->nb[1], x->nb[2], x->nb[3], x->nb[1] * x->nb[2] * x->nb[3]);
    struct ggml_tensor* r = ggml_view_4d(ctx, x, batch_size, 1, seq_len, model->hparams.dim, x->nb[1], x->nb[2], x->nb[3], 2 * x->nb[2]);
    struct ggml_tensor* s = ggml_view_4d(ctx, x, batch_size, 1, seq_len, model->hparams.dim, x->nb[1], x->nb[2], x->nb[3], 3 * x->nb[2]);

    struct ggml_tensor* _x = ggml_map_custom2(ctx, p, ggml_cast(ctx, model->embed0_w, GGML_TYPE_F32), magnet_embedding, GGML_N_TASKS_MAX, NULL);
    _x = ggml_add(ctx, _x, ggml_map_custom2(ctx, q, ggml_cast(ctx, model->embed1_w, GGML_TYPE_F32), magnet_embedding, GGML_N_TASKS_MAX, NULL));
    _x = ggml_add(ctx, _x, ggml_map_custom2(ctx, r, ggml_cast(ctx, model->embed2_w, GGML_TYPE_F32), magnet_embedding, GGML_N_TASKS_MAX, NULL));
    _x = ggml_add(ctx, _x, ggml_map_custom2(ctx, s, ggml_cast(ctx, model->embed3_w, GGML_TYPE_F32), magnet_embedding, GGML_N_TASKS_MAX, NULL));

    return _x;
}

ggml_cgraph* build_graph(magnet_model& model, struct ggml_tallocr* allocr)
{
    static size_t buf_size = ggml_tensor_overhead() * 10000 + ggml_graph_overhead() + (1024 * 1024 * 1024);
    static std::vector<uint8_t> buf(buf_size);

    // create dummy context
    struct ggml_init_params params0 = {
        .mem_size = buf_size,
        .mem_buffer = buf.data(),
        .no_alloc = true
    };

    // create temporary context to build the grpah
    struct ggml_context* ctx0 = ggml_init(params0);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 3072, false);

    // The transformer expects S, E (sequence len, embed_dim)
    // total forward expects (T, C) (tokens, channels) not there yet
    //struct ggml_tensor* input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 4, model.hparams.dim);
    struct ggml_tensor* input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, 4, 5, 1024);
    struct ggml_tensor* cross_attn_src = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 4, model.hparams.dim);
    ggml_tallocr_alloc(allocr, input);
    ggml_tallocr_alloc(allocr, cross_attn_src);

    input = ggml_set_f32(input, 1.0f);
    input = ggml_map_custom1(ctx0, input, magnet_test_fill, GGML_N_TASKS_MAX, NULL);
    cross_attn_src = ggml_set_f32(cross_attn_src, 1.0f);

    //struct ggml_tensor* result = magnet_transformer_forward(&model, ctx0, allocr, input, cross_attn_src);
    struct ggml_tensor* result = magnet_forward(&model, ctx0, allocr, input);
    //struct ggml_tensor* result = input;

    ggml_build_forward_expand(gf, result);

    ggml_free(ctx0);
    return gf;
}

int main(int argc, char** argv)
{
    std::string file_name = "/home/cat/src/magnet.cpp/mdl/small/ggml_model.bin";
    if (argc != 2) {
        fprintf(stderr, "%s: File path argument not provided\n", __func__);
        return -1;
    }
    file_name = std::string(argv[1]);

    magnet_context* magnet_ctx = new magnet_context();
    magnet_ctx->model = magnet_model();

    if (!load_parameters(file_name, magnet_ctx->model)) {
        fprintf(stderr, "%s: Failed to load model parameters\n", __func__);
        return -1;
    }

    auto& ctx = magnet_ctx->model.ctx;
    GGML_ASSERT(ctx != nullptr);
    auto& model = magnet_ctx->model;

    magnet_ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    model.buffer = ggml_backend_alloc_buffer(model.backend, 1024 * 1024 * 1024); // fuck it, give it a gig!
    magnet_ctx->talloc = ggml_tallocr_new(model.buffer);

    ggml_cgraph* graph = build_graph(model, &magnet_ctx->talloc);
    ggml_gallocr_alloc_graph(magnet_ctx->galloc, graph);
    ggml_graph_print(graph);

    if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        ggml_free(magnet_ctx->model.ctx);
        ggml_backend_free(model.backend);
        ggml_gallocr_free(magnet_ctx->galloc);
        delete magnet_ctx;
        return -1;
    }

    auto out = graph->nodes[graph->n_nodes - 1];
    
    printf("\n\nemb tensor:");
    print_tensor(graph->leafs[1]);

    printf("\n\nout tensor:");
    print_tensor(out);

    ggml_free(magnet_ctx->model.ctx);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(magnet_ctx->galloc);
    delete magnet_ctx;
    return 0;
}
