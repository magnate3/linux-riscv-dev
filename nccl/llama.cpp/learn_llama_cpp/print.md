#  print

```
 llama_batch_print
 llama_memory_breakdown_print(ctx); // goes to debug log
```
+   llama_batch_print   
```
static void llama_batch_print(const llama_batch *batch) {
  printf("%s\n", std::string(20, '-').c_str());
  printf("%30s: %-10d\n", "n_tokens", batch->n_tokens);

  printf("tokens|emb:\n");
  for (int i = 0; i < batch->n_tokens && batch->token; i++)
    printf("%8d,", batch->token[i]);
  for (int i = 0; i < batch->n_tokens && batch->embd; i++)
    printf("%8f,", batch->embd[i]);
  printf("\npos: \n");
  for (int i = 0; i < batch->n_tokens && batch->pos; i++)
    printf("%8d,", batch->pos[i]);
  printf("\nn_seq\n");
  for (int i = 0; i < batch->n_tokens && batch->seq_id[i]; i++)
    printf("%8d,", batch->seq_id[i][0]);

  printf("\n");
};
```



```
void llama_batch_allocr::ubatch_print(const llama_ubatch & ubatch, int debug) {
    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s:   equal_seqs   = %d\n", __func__, ubatch.equal_seqs());
        LLAMA_LOG_DEBUG("%s:   n_tokens     = %d\n", __func__, ubatch.n_tokens);
        LLAMA_LOG_DEBUG("%s:   n_seq_tokens = %d\n", __func__, ubatch.n_seq_tokens);
        LLAMA_LOG_DEBUG("%s:   n_seqs       = %d\n", __func__, ubatch.n_seqs);
        LLAMA_LOG_DEBUG("%s:   n_seqs_unq   = %d\n", __func__, ubatch.n_seqs_unq);

        std::stringstream ss_seq_id_unq;
        std::stringstream ss_seq_idx;

        ss_seq_id_unq << "[ ";
        ss_seq_idx << "[";

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            ss_seq_id_unq << ubatch.seq_id_unq[s] << " ";
        }

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (ubatch.seq_idx[s] >= 0) {
                ss_seq_idx << ubatch.seq_idx[s]%10;
            } else {
                ss_seq_idx << ".";
            }
        }

        ss_seq_id_unq << "]";
        ss_seq_idx    << "]";

        LLAMA_LOG_DEBUG("%s:   token      = %p\n", __func__, (void *) ubatch.token);
        LLAMA_LOG_DEBUG("%s:   embd       = %p\n", __func__, (void *) ubatch.embd);
        LLAMA_LOG_DEBUG("%s:   pos        = %p\n", __func__, (void *) ubatch.pos);
        LLAMA_LOG_DEBUG("%s:   n_seq_id   = %p\n", __func__, (void *) ubatch.n_seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id     = %p\n", __func__, (void *) ubatch.seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id_unq = %s\n", __func__, ss_seq_id_unq.str().c_str());
        LLAMA_LOG_DEBUG("%s:   seq_idx    = %s\n", __func__, ss_seq_idx.str().c_str());
        LLAMA_LOG_DEBUG("%s:   output     = %p\n", __func__, (void *) ubatch.output);
        LLAMA_LOG_DEBUG("%s:   n_outputs  = %d\n", __func__, n_outputs);

        if (debug > 1) {
            int seq_id_max = 0;
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                        seq_id_max = std::max(seq_id_max, ubatch.seq_id[i][s]);
                    }
                }
            }
            ++seq_id_max;

            LLAMA_LOG_DEBUG("%s:   token     = [\n", __func__);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                std::vector<int8_t> seq_id(seq_id_max);

                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    seq_id[ubatch.seq_id[i][s]] = 1;
                }

                std::stringstream ss;
                for (int s = 0; s < seq_id_max; ++s) {
                    if (seq_id[s]) {
                        ss << s%10;
                    } else {
                        ss << ".";
                    }
                }

                if (ubatch.token) {
                    LLAMA_LOG_DEBUG("%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.token[i], vocab->token_to_piece(ubatch.token[i]).c_str(),
                            ubatch.pos[i], ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                } else {
                    LLAMA_LOG_DEBUG("%s:  %4d: [embd], pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.pos[i], ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                }
            }
            LLAMA_LOG_DEBUG("%s:   ]\n", __func__);
        }
    }
}
```

```
const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}
```

# kv cache


```
//
// kv cache
//

// deprecated
int32_t llama_kv_self_n_tokens(const llama_context * ctx) {
    const auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return 0;
    }

    int32_t res = 0;

    for (uint32_t s = 0; s < ctx->get_cparams().n_seq_max; s++) {
        const llama_pos p0 = kv->seq_pos_min(s);
        const llama_pos p1 = kv->seq_pos_max(s);

        if (p0 >= 0) {
            res += (p1 - p0) + 1;
        }
    }

    return res;
}

// deprecated
// note: this is the same as above - will be removed anyway, so it's ok
int32_t llama_kv_self_used_cells(const llama_context * ctx) {
    const auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return 0;
    }

    int32_t res = 0;

    for (uint32_t s = 0; s < ctx->get_cparams().n_seq_max; s++) {
        const llama_pos p0 = kv->seq_pos_min(s);
        const llama_pos p1 = kv->seq_pos_max(s);

        if (p0 >= 0) {
            res += (p1 - p0) + 1;
        }
    }

    return res;
}

// deprecated
void llama_kv_self_clear(llama_context * ctx) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return;
    }

    llama_memory_clear(kv, true);
}

// deprecated
bool llama_kv_self_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return true;
    }

    return llama_memory_seq_rm(kv, seq_id, p0, p1);
}

// deprecated
void llama_kv_self_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return;
    }

    llama_memory_seq_cp(kv, seq_id_src, seq_id_dst, p0, p1);
}

// deprecated
void llama_kv_self_seq_keep(llama_context * ctx, llama_seq_id seq_id) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return;
    }

    llama_memory_seq_keep(kv, seq_id);
}

// deprecated
void llama_kv_self_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return;
    }

    llama_memory_seq_add(kv, seq_id, p0, p1, delta);
}

// deprecated
void llama_kv_self_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return;
    }

    llama_memory_seq_div(kv, seq_id, p0, p1, d);
}

// deprecated
llama_pos llama_kv_self_seq_pos_min(llama_context * ctx, llama_seq_id seq_id) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return -1;
    }

    return llama_memory_seq_pos_min(kv, seq_id);
}

// deprecated
llama_pos llama_kv_self_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return -1;
    }

    return llama_memory_seq_pos_max(kv, seq_id);
}

// deprecated
void llama_kv_self_defrag(llama_context * ctx) {
    // force defrag
    ctx->kv_self_defrag_sched();
}

// deprecated
bool llama_kv_self_can_shift(const llama_context * ctx) {
    auto * kv = llama_get_memory(ctx);
    if (!kv) {
        return false;
    }

    return llama_memory_can_shift(kv);
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

```