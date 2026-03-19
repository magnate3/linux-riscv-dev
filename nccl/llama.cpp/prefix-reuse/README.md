


```
    // === Incremental Inference Logic ===
        // 检测 tools 是否变化（变化则需要完全重新 prefill）
        const size_t new_tools_hash = hash_string(tools_str);
        const bool tools_changed = (new_tools_hash != g_last_tools_hash);

        // 计算公共前缀长度
        size_t n_past = 0;
        if (!tools_changed && !g_last_prompt_tokens_vec.empty()) {
            n_past = get_common_prefix(g_last_prompt_tokens_vec, new_tokens);
        }

        // 如果 tools 变化或没有公共前缀，完全重新开始
        if (tools_changed || n_past == 0) {
            LOG_DIAGi("%s: Full prefill (tools_changed=%d, n_past=%zu)",
                      __func__, tools_changed, n_past);
            reset_long_term_states(true);
            n_past = 0;
        } else if (n_past < (size_t) current_position) {
            // 清除 [n_past, end) 的 KV Cache
            LOG_DIAGi("%s: Incremental prefill: clearing KV cache from %zu to %d",
                      __func__, n_past, current_position);
            llama_memory_seq_rm(llama_get_memory(g_context), 0, (llama_pos) n_past, current_position);
            current_position = (llama_pos) n_past;
        }

        // 只 Prefill 新增的 token
        const size_t n_new = new_tokens.size() - n_past;
        if (n_new > 0) {
            llama_tokens new_part(new_tokens.begin() + n_past, new_tokens.end());
            if (decode_tokens_in_batches(g_context, g_batch, new_part, current_position, true)) {
                LOG_DIAGe("%s: llama_decode() failed during prompt prefill", __func__);
                return 3;
            }
            current_position += (int) n_new;
        }

        LOG_DIAGi("%s: prefill total=%zu reused=%zu new=%zu",
                  __func__, new_tokens.size(), n_past, n_new);

        // 记录复用的 token 数
        g_last_reused_tokens = n_past;

        // 记录实际 prefill 的 token 数（不包括复用的）
        g_last_prompt_tokens = (int) n_new;

        // 更新状态
        g_last_prompt_tokens_vec = std::move(new_tokens);
        g_last_tools_hash = new_tools_hash;
    }

    stop_generation_position = current_position + DEFAULT_MAX_NEW_TOKENS;

    rebuild_sampler(&g_last_chat_params);
    common_sampler_reset(g_sampler);

    return 0;
```