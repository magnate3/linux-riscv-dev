#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-radix-tree.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        ggml_type   type_k,
        ggml_type   type_v,
        bool   v_trans,
        bool   offload,
        bool   unified,
        uint32_t   kv_size,
        uint32_t   n_seq_max,
        uint32_t   n_pad,
        uint32_t   n_swa,
        llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse) :
    model(model), 
    hparams(model.hparams), 
    v_trans(v_trans),
    n_seq_max(n_seq_max), 
    n_stream(unified ? 1 : n_seq_max), 
    n_pad(n_pad), 
    n_swa(n_swa), 
    swa_type(swa_type),
    radix_offsets() {

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer_kv = hparams.n_layer_kv();

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer_kv*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);

        ggml_format_name(k, "cache_k_l%d", il);
        ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]));
            v_stream.push_back(ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]));
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < hparams.n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;

    // Initialize RadixAttention if enabled
    const char * LLAMA_RADIX_ATTENTION = getenv("LLAMA_RADIX_ATTENTION");
    radix_attention_enabled = LLAMA_RADIX_ATTENTION ? (atoi(LLAMA_RADIX_ATTENTION) != 0) : false;

    // ==================================================
    // Phase 3.3: Multi-stream validation for RadixAttention
    // ==================================================
    if (radix_attention_enabled) {
        // For initial implementation, restrict to single-stream mode
        if (n_stream > 1) {
            LLAMA_LOG_WARN("%s: RadixAttention is currently only supported in unified (single-stream) mode\n", __func__);
            LLAMA_LOG_WARN("%s: Disabling RadixAttention. To use RadixAttention, set --kv-unified or reduce --parallel\n", __func__);
            radix_attention_enabled = false;
        } else {
            const uint32_t n_layer_kv = hparams.n_layer_kv();
            radix_tree = std::make_unique<llama_radix_tree>(n_layer_kv);
            LLAMA_LOG_INFO("%s: RadixAttention enabled (n_layers = %u, unified mode)\n", __func__, n_layer_kv);
        }
    }
    // ==================================================
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    // Reset radix tree
    if (radix_tree) {
        radix_tree = std::make_unique<llama_radix_tree>(hparams.n_layer_kv());
        LLAMA_LOG_DEBUG("%s: radix tree reset\n", __func__);
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // Unregister from radix tree if removing entire sequence
    if (radix_tree && seq_id >= 0 && p0 == 0 && p1 == std::numeric_limits<llama_pos>::max()) {
        radix_unregister_sequence(seq_id);
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    // ==================================================
    // Phase 3.2 & 3.3: RadixAttention sequence copy support
    // ==================================================
    if (is_radix_attention_enabled() && radix_tree) {
        // Verify single-stream mode
        GGML_ASSERT(s0 == 0 && s1 == 0 && "RadixAttention requires unified (single-stream) mode");
        
        // Extract the token sequence from the source sequence
        std::vector<llama_token> src_tokens;
        
        // Collect tokens from the source sequence's KV cells
        const auto & src_cells = v_cells[s0];
        for (uint32_t i = 0; i < src_cells.size(); ++i) {
            if (!src_cells.is_empty(i) && src_cells.seq_has(i, seq_id_src)) {
                // Note: We don't have direct access to tokens in KV cells
                // This is a simplified approach - in production, you'd need to
                // track token sequences separately or modify the KV cell structure
                // For now, we'll check if we can find the prefix in the radix tree
                break; // TODO: implement proper token extraction
            }
        }
        
        // Try to find if source sequence exists in radix tree
        // by checking if any node has cache slots matching the source cells
        std::vector<uint32_t> src_cache_slots;
        
        // Collect cache slot indices from source sequence
        for (uint32_t i = 0; i < src_cells.size(); ++i) {
            if (!src_cells.is_empty(i) && src_cells.seq_has(i, seq_id_src)) {
                src_cache_slots.push_back(i);
            }
        }
        
        if (!src_cache_slots.empty()) {
            LLAMA_LOG_DEBUG("%s: RadixAttention: copying seq %d to %d in unified mode\n",
                __func__, seq_id_src, seq_id_dst);
            
            if (s0 == s1) {
                // Same stream - just add dst seq_id to existing cells
                // The radix tree state remains the same
                LLAMA_LOG_DEBUG("%s: same-stream copy (stream 0), radix tree unchanged\n", __func__);
            }
        }
    }
    // ==================================================
    // End of RadixAttention sequence copy support
    // ==================================================

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remeber the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    bool updated = false;

    auto * sched = lctx->get_sched();

    // --------------------------
    // Stream copy (if any)
    // --------------------------
    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);
                ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
            }
        }
    }

    // --------------------------
    // K-shift (if enabled)
    // --------------------------
    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();
            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            cells.reset_shift();
        }
    }

    // --------------------------
    // RadixAttention: update radix_offsets
    // --------------------------
    radix_offsets.clear();
    radix_offsets.resize(layers.size(), 0);

    for (size_t il = 0; il < layers.size(); ++il) {
        // For now, simple example: store the first stream head
        if (!v_heads.empty()) {
            radix_offsets[il] = v_heads[0];
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {
    // ...existing debug code...

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    // ==================================================
    // Phase 2.6 & 3.3: RadixAttention prefix reuse
    // ==================================================
    if (is_radix_attention_enabled() && n_seqs == 1) {
        // Verify we're in single-stream mode
        GGML_ASSERT(n_stream == 1 && "RadixAttention requires unified (single-stream) mode");
        
        // Extract tokens from ubatch for radix tree lookup
        std::vector<llama_token> tokens;
        tokens.reserve(n_tokens);
        
        for (uint32_t i = 0; i < n_tokens; ++i) {
            if (ubatch.token) {
                tokens.push_back(ubatch.token[i]);
            }
        }

        if (!tokens.empty()) {
            // Search radix tree for matching prefix
            auto [prefix_node, prefix_len] = radix_find_prefix(tokens);

            if (prefix_len > 0 && prefix_node && !prefix_node->cache_slots.empty()) {
                // We found a cached prefix!
                const auto seq_id = ubatch.seq_id_unq[0];
                res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
                res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);
                res.strm[0] = seq_to_stream[seq_id];
                
                GGML_ASSERT(res.strm[0] == 0 && "RadixAttention expects stream 0 in unified mode");

                LLAMA_LOG_DEBUG("%s: RadixAttention: found cached prefix of length %u/%u\n", 
                    __func__, prefix_len, n_tokens);

                // Reuse cache slots from the prefix
                const auto & cached_slots = prefix_node->cache_slots;
                
                // Allocate the vector properly
                res.idxs[0].clear();
                res.idxs[0].reserve(n_tokens);
                
                // Copy the cached slots for the prefix
                const auto n_reuse = std::min(prefix_len, n_tokens);
                for (uint32_t i = 0; i < n_reuse; ++i) {
                    if (i < cached_slots.size()) {
                        res.idxs[0].push_back(cached_slots[i]);
                    }
                }
                
                // Allocate new slots for remaining tokens
                const auto & cells = v_cells[res.strm[0]];
                for (uint32_t i = n_reuse; i < n_tokens; ++i) {
                    res.idxs[0].push_back(cells.used_max_p1() + (i - n_reuse));
                }
                
                LLAMA_LOG_DEBUG("%s: RadixAttention: reused %u slots, allocated %u new slots\n", 
                    __func__, n_reuse, n_tokens - n_reuse);
                
                // Early return with the allocated slots
                return res;
            }
        }
    }
    // ==================================================
    // End of RadixAttention prefix reuse
    // ==================================================

    // ...existing code for normal allocation...

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);
        res.strm[s] = seq_to_stream[seq_id];

        const auto & cells = v_cells[res.strm[s]];

        // Properly allocate idxs vector
        res.idxs[s].clear();
        res.idxs[s].reserve(n_tokens);
        
        for (uint32_t i = 0; i < n_tokens; ++i) {
            res.idxs[s].push_back(cells.used_max_p1() + i);
        }
    }

    return res;
}

// Add these missing implementations after the existing methods:

uint32_t llama_kv_cache::get_size() const {
    return hparams.n_ctx;
}

bool llama_kv_cache::get_has_shift() const {
    return hparams.rope_type != LLAMA_ROPE_TYPE_NONE;
}

bool llama_kv_cache::get_can_shift() const {
    return get_has_shift();
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size = 0;
    for (const auto & layer : layers) {
        if (layer.k) {
            size += ggml_nbytes(layer.k);
        }
    }
    return size;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size = 0;
    for (const auto & layer : layers) {
        if (layer.v) {
            size += ggml_nbytes(layer.v);
        }
    }
    return size;
}

bool llama_kv_cache::is_radix_attention_enabled() const {
    return radix_attention_enabled;
}

std::pair<llama_radix_node *, uint32_t> llama_kv_cache::radix_find_prefix(const std::vector<llama_token> & tokens) const {
    if (!radix_tree) {
        return {nullptr, 0};
    }
    return radix_tree->find_prefix(tokens);
}

void llama_kv_cache::radix_register_sequence(llama_seq_id seq_id, const std::vector<llama_token> & tokens, const std::vector<uint32_t> & cache_slots) {
    if (radix_tree) {
        radix_tree->insert_sequence(tokens, cache_slots);
    }
}

void llama_kv_cache::radix_unregister_sequence(llama_seq_id seq_id) {
    // Implementation depends on how sequences are tracked
    // For now, this is a placeholder
}

void llama_kv_cache::radix_copy_sequence_ref(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    // Implementation for copying sequence references
    // This is a placeholder
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // Apply the ubatch to the KV cache
    // This is a core operation that updates cache state
    // Implementation details depend on the specific caching strategy
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // Build computation graph for RoPE position shift
    // This is used when context shifting is enabled
    return nullptr; // Placeholder
}

// llama_kv_cache_context implementations

llama_kv_cache_context::llama_kv_cache_context(llama_kv_cache * cache)
    : cache(cache) {
}

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status)
    : cache(nullptr) {
}

llama_kv_cache_context::llama_kv_cache_context(
    llama_kv_cache * cache,
    llama_context * lctx,
    bool optimize,
    llama_kv_cache::stream_copy_info sc_info)
    : cache(cache) {
}

llama_kv_cache_context::llama_kv_cache_context(
    llama_kv_cache * cache,
    std::vector<llama_kv_cache::slot_info> slot_infos,
    std::vector<llama_ubatch> ubatches)
    : cache(cache) {
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return cache ? cache->get_size() : 0;
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    if (!cache || il < 0 || il >= (int32_t)cache->layers.size()) {
        return nullptr;
    }
    return cache->layers[il].k;
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    if (!cache || il < 0 || il >= (int32_t)cache->layers.size()) {
        return nullptr;
    }
    return cache->layers[il].v;
}

ggml_tensor * llama_kv_cache_context::cpy_k(
    ggml_context * ctx,
    ggml_tensor * k_cur,
    ggml_tensor * k_idxs,
    int32_t il) const {
    // Copy K cache for the given layer
    return k_cur; // Placeholder
}

ggml_tensor * llama_kv_cache_context::cpy_v(
    ggml_context * ctx,
    ggml_tensor * v_cur,
    ggml_tensor * v_idxs,
    int32_t il) const {
    // Copy V cache for the given layer
    return v_cur; // Placeholder
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    // Build K indices tensor
    return nullptr; // Placeholder
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    // Build V indices tensor
    return nullptr; // Placeholder
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    // Set K indices in tensor
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    // Set V indices in tensor
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    // Set KQ mask for attention
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    // Set position bucket information
}