
#  sample_tree


```
ctx_params.kv_unified =  true;
```


```

./build/sample_tree   -m /workspace/qwen/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "Hello my name is" -np 4 
```

```
main: n_predict = 32, n_ctx = 16384, n_batch = 512, n_parallel = 4, n_kv_req = 113

<|begin_of_text|>Hello my name is cells seq count 4      cells pos[0] pos_in 0-5         cells seq[0] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[1] pos_in 0-5         cells seq[1] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[2] pos_in 0-5         cells seq[2] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[3] pos_in 0-5         cells seq[3] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[4] pos_in 0-5         cells seq[4] has stream 0  ,1  ,2  ,3 
min[0] =     0, max[0] =     4
min[1] =     0, max[1] =     4
min[2] =     0, max[2] =     4
min[3] =     0, max[3] =     4


main: generating 4 sequences ...
```

```
main: stream 3 finished at n_cur = 32
main: stream 3 finished at n_cur = 32 ,2  ,3 
 cells seq count 4       cells pos[1] pos_in 0-32        cells seq[1] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[2] pos_in 0-32        cells seq[2] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[3] pos_in 0-32        cells seq[3] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[4] pos_in 0-32        cells seq[4] has stream 0  ,1  ,2  ,3 
 cells seq count 1       cells pos[5] pos_in 0-32        cells seq[5] has stream 0 
 cells seq count 1       cells pos[6] pos_in 0-32        cells seq[6] has stream 0 
 cells seq count 1       cells pos[7] pos_in 0-32        cells seq[7] has stream 0 
 cells seq count 3       cells pos[8] pos_in 0-32        ,1  ,2  ,3 
 cells seq count 1       cells pos[9] pos_in 0-32        cells seq[9] has stream 0 
 cells seq count 1       cells pos[10] pos_in 0-32       cells seq[10] has stream 0 
 cells seq count 1       cells pos[11] pos_in 0-32       cells seq[11] has stream 0 
 cells seq count 3       cells pos[12] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[13] pos_in 0-32       cells seq[13] has stream 0 
 cells seq count 3       cells pos[14] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[15] pos_in 0-32       cells seq[15] has stream 0 
 cells seq count 3       cells pos[16] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[17] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[18] pos_in 0-32       cells seq[18] has stream 0 
 cells seq count 1       cells pos[19] pos_in 0-32       cells seq[19] has stream 0 
 cells seq count 1       cells pos[20] pos_in 0-32       cells seq[20] has stream 0 
 cells seq count 3       cells pos[21] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[22] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[23] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[24] pos_in 0-32       cells seq[24] has stream 0 
 cells seq count 1       cells pos[25] pos_in 0-32       cells seq[25] has stream 0 
 cells seq count 1       cells pos[26] pos_in 0-32       cells seq[26] has stream 0 
 cells seq count 3       cells pos[27] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[28] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[29] pos_in 0-32       cells seq[29] has stream 0 
 cells seq count 1       cells pos[30] pos_in 0-32       cells seq[30] has stream 0 
 cells seq count 3       cells pos[31] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[32] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[33] pos_in 0-32       cells seq[33] has stream 0 
 cells seq count 3       cells pos[34] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[35] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[36] pos_in 0-32       cells seq[36] has stream 0 
 cells seq count 1       cells pos[37] pos_in 0-32       cells seq[37] has stream 0 
 cells seq count 3       cells pos[39] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[40] pos_in 0-32       cells seq[40] has stream 0 
 cells seq count 1       cells pos[41] pos_in 0-32       cells seq[41] has stream 0 
 cells seq count 3       cells pos[42] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[43] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[44] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[45] pos_in 0-32       cells seq[45] has stream 0 
 cells seq count 1       cells pos[46] pos_in 0-32       cells seq[46] has stream 0 
 cells seq count 1       cells pos[47] pos_in 0-32       ,2 
 cells seq count 3       cells pos[48] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[49] pos_in 0-32       cells seq[49] has stream 0 
 cells seq count 1       cells pos[50] pos_in 0-32       ,2 
 cells seq count 2       cells pos[51] pos_in 0-32       ,1  ,3 
 cells seq count 3       cells pos[52] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[53] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[54] pos_in 0-32       cells seq[54] has stream 0 
 cells seq count 1       cells pos[55] pos_in 0-32       ,2 
 cells seq count 3       cells pos[56] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[57] pos_in 0-32       cells seq[57] has stream 0 
 cells seq count 1       cells pos[58] pos_in 0-32       cells seq[58] has stream 0 
 cells seq count 2       cells pos[60] pos_in 0-32       ,1  ,3 
 cells seq count 3       cells pos[61] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[62] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[65] pos_in 0-32       ,1  ,2  ,3 

```

## n_stream and ubatch.n_seqs_unq


```
const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
```

```
./build/sample_tree_kv_dbg   -m /workspace/qwen/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "Hello my name is" -np 4
```


```
ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

```

```
static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        const auto n_kv     = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;
        const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    return inp;
}
```

```
void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

```

```
print_ggml_kv: backend name CPU, tensor cache_k_l15 (view)
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 0, llama pos 0 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 1, llama pos 1 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 2, llama pos 2 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 3, llama pos 3 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 4, llama pos 4 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 5, llama pos 5 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 6, llama pos 8 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 9, llama pos 6 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 0 cells index: 0,ggml data pos base 0 and offset 13, llama pos 7 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 0, llama pos 0 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 1, llama pos 1 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 2, llama pos 2 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 3, llama pos 3 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 4, llama pos 4 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 5, llama pos 5 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 9, llama pos 6 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 10, llama pos 8 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 1 cells index: 0,ggml data pos base 512 and offset 13, llama pos 7 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 0, llama pos 0 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 1, llama pos 1 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 2, llama pos 2 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 3, llama pos 3 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 4, llama pos 4 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 7, llama pos 5 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 11, llama pos 6 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 14, llama pos 8 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 2 cells index: 0,ggml data pos base 1024 and offset 15, llama pos 7 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 0, llama pos 0 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 1, llama pos 1 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 2, llama pos 2 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 3, llama pos 3 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 4, llama pos 4 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 8, llama pos 5 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 12, llama pos 6 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 16, llama pos 7 
n_stream 1,ubatch.n_seqs_unq 4, seq_id 3 cells index: 0,ggml data pos base 1536 and offset 17, llama pos 8 
 
```


##  error

```
common_batch_add(batch, id, n_cur+i, { s }, true);
```

```
init: the tokens of sequence 1 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 1 is X = 4
 - the tokens for sequence 1 in the input batch have a starting position of Y = 6
 it is required that the sequence positions remain consecutive: Y = X + 1
decode: failed to initialize batch
llama_decode: failed to decode, ret = -1
main : failed to eval, return code 1
```