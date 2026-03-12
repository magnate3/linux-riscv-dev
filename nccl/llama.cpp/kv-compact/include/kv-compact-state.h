// KV Cache State Buffer Parser and Writer
//
// Parses the binary format produced by llama_state_seq_get_data() into
// structured per-layer, per-head K/V data (as float32), and writes
// compacted state buffers back in the same format.
//
// State format (per stream):
//   [n_stream:u32]
//   per stream:
//     [cell_count:u32]
//     per cell: [pos:i32] [n_seq_id:u32] [seq_ids:i32*n_seq_id]
//     [v_trans:u32] [n_layer:u32]
//     per layer: [k_type:i32] [k_size_row:u64] [k_data:u8*cell_count*k_size_row]
//     per layer (non-trans): [v_type:i32] [v_size_row:u64] [v_data:u8*cell_count*v_size_row]
//     per layer (trans): [v_type:i32] [v_size_el:u32] [n_embd_v_gqa:u32] [v_data:u8*n_embd_v_gqa*cell_count*v_size_el]

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Forward declare ggml types we need (avoid full ggml.h dependency in header)
#ifndef GGML_TYPE_F32
#define KV_COMPACT_GGML_TYPE_F32  0
#define KV_COMPACT_GGML_TYPE_F16  1
#else
#define KV_COMPACT_GGML_TYPE_F32  GGML_TYPE_F32
#define KV_COMPACT_GGML_TYPE_F16  GGML_TYPE_F16
#endif

// ============================================================================
// Cell metadata
// ============================================================================

struct kv_cell_meta {
    int32_t  pos;
    int32_t  ext_x = 0;  // IMROPE: x position (M-RoPE spatial)
    int32_t  ext_y = 0;  // IMROPE: y position (M-RoPE spatial)
    std::vector<int32_t> seq_ids;
};

// ============================================================================
// Parsed layer data — all heads concatenated, converted to float32
// ============================================================================

struct kv_layer_data {
    int32_t  k_type;        // original ggml_type
    int32_t  v_type;
    uint64_t k_size_row;    // bytes per row in original format
    uint64_t v_size_row;    // bytes per row (non-transposed only)
    uint32_t v_size_el;     // bytes per element (transposed only)
    uint32_t n_embd_v_gqa;  // V embedding size (transposed only)

    // Float32 data: [cell_count, n_embd_k_gqa] and [cell_count, n_embd_v_gqa]
    // V is always stored as [cell_count, n_embd_v_gqa] regardless of v_trans
    std::vector<float> K;   // all heads concatenated, row = token
    std::vector<float> V;   // all heads concatenated, row = token (transposed from storage if needed)

    int n_embd_k_gqa() const { return K.empty() ? 0 : (int)(K.size() / cell_count); }
    int n_embd_v_gqa_computed() const { return V.empty() ? 0 : (int)(V.size() / cell_count); }

    uint32_t cell_count = 0;  // set during parsing
};

// ============================================================================
// Parsed KV state
// ============================================================================

struct parsed_kv_state {
    uint32_t n_stream = 0;

    // Per-stream data (usually just 1 stream for seq_id=0)
    struct stream_data {
        uint32_t cell_count = 0;
        std::vector<kv_cell_meta> cells;
        uint32_t v_trans = 0;
        uint32_t n_layer = 0;
        std::vector<kv_layer_data> layers;
    };
    std::vector<stream_data> streams;

    // Raw state buffer (kept for round-trip; overwritten by write_state)
    std::vector<uint8_t> raw;

    // Trailing data after KV section (e.g., recurrent section for hybrid SSM+MoE models)
    std::vector<uint8_t> trailing_data;

    // ---- Parsing ----

    // n_pos_per_embd: 1 for normal RoPE, 4 for M-RoPE/IMROPE
    //   When > 1, each cell has extra 8 bytes (llama_kv_cell_ext: x,y positions)
    //   after n_seq_id and before seq_ids
    bool parse(const uint8_t * data, size_t size, uint32_t n_pos_per_embd = 1) {
        raw.assign(data, data + size);

        const uint8_t * ptr = data;
        const uint8_t * end = data + size;

        if (!read_val(ptr, end, n_stream)) return false;
        streams.resize(n_stream);

        for (uint32_t s = 0; s < n_stream; s++) {
            auto & sd = streams[s];
            if (!read_val(ptr, end, sd.cell_count)) return false;

            if (sd.cell_count == 0) continue;

            // Parse cell metadata
            sd.cells.resize(sd.cell_count);
            for (uint32_t c = 0; c < sd.cell_count; c++) {
                auto & cell = sd.cells[c];
                if (!read_val(ptr, end, cell.pos)) return false;
                uint32_t n_seq_id;
                if (!read_val(ptr, end, n_seq_id)) return false;
                // Read llama_kv_cell_ext (x,y) for M-RoPE/IMROPE models
                if (n_pos_per_embd > 1) {
                    if (!read_val(ptr, end, cell.ext_x)) return false;
                    if (!read_val(ptr, end, cell.ext_y)) return false;
                }
                cell.seq_ids.resize(n_seq_id);
                for (uint32_t i = 0; i < n_seq_id; i++) {
                    if (!read_val(ptr, end, cell.seq_ids[i])) return false;
                }
            }

            // v_trans and n_layer
            if (!read_val(ptr, end, sd.v_trans)) return false;
            if (!read_val(ptr, end, sd.n_layer)) return false;

            sd.layers.resize(sd.n_layer);

            // Parse K data per layer
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                auto & ld = sd.layers[l];
                ld.cell_count = sd.cell_count;

                if (!read_val(ptr, end, ld.k_type)) return false;
                if (!read_val(ptr, end, ld.k_size_row)) return false;

                const size_t k_data_size = sd.cell_count * ld.k_size_row;
                if (ptr + k_data_size > end) return false;

                // Convert to float32
                const int n_floats_per_row = (int)(ld.k_size_row / type_size(ld.k_type));
                ld.K.resize((size_t)sd.cell_count * n_floats_per_row);
                convert_to_f32(ptr, ld.k_type, ld.K.data(), sd.cell_count * n_floats_per_row);

                ptr += k_data_size;
            }

            // Parse V data per layer
            if (!sd.v_trans) {
                for (uint32_t l = 0; l < sd.n_layer; l++) {
                    auto & ld = sd.layers[l];

                    if (!read_val(ptr, end, ld.v_type)) return false;
                    if (!read_val(ptr, end, ld.v_size_row)) return false;

                    const size_t v_data_size = sd.cell_count * ld.v_size_row;
                    if (ptr + v_data_size > end) return false;

                    const int n_floats_per_row = (int)(ld.v_size_row / type_size(ld.v_type));
                    ld.V.resize((size_t)sd.cell_count * n_floats_per_row);
                    convert_to_f32(ptr, ld.v_type, ld.V.data(), sd.cell_count * n_floats_per_row);

                    ptr += v_data_size;
                }
            } else {
                // Transposed V
                for (uint32_t l = 0; l < sd.n_layer; l++) {
                    auto & ld = sd.layers[l];

                    if (!read_val(ptr, end, ld.v_type)) return false;
                    if (!read_val(ptr, end, ld.v_size_el)) return false;
                    if (!read_val(ptr, end, ld.n_embd_v_gqa)) return false;

                    const size_t v_data_size = (size_t)ld.n_embd_v_gqa * sd.cell_count * ld.v_size_el;
                    if (ptr + v_data_size > end) return false;

                    // Transpose from [embd][token] to [token][embd]
                    ld.V.resize((size_t)sd.cell_count * ld.n_embd_v_gqa);
                    transpose_v_to_f32(ptr, ld.v_type, ld.V.data(),
                                       sd.cell_count, ld.n_embd_v_gqa);

                    ptr += v_data_size;
                }
            }
        }

        // Save any remaining data (e.g., recurrent section for hybrid SSM+MoE models)
        if (ptr < end) {
            trailing_data.assign(ptr, end);
        }

        return true;
    }

    // ---- Extract per-head data ----

    // Get K for a specific head: output [cell_count, d_k]
    void get_k_head(int stream, int layer, int head, int d_k, std::vector<float> & out) const {
        const auto & ld = streams[stream].layers[layer];
        const int n_embd = ld.n_embd_k_gqa();
        const int cc = ld.cell_count;
        out.resize(cc * d_k);
        for (int i = 0; i < cc; i++) {
            memcpy(out.data() + i * d_k,
                   ld.K.data() + i * n_embd + head * d_k,
                   d_k * sizeof(float));
        }
    }

    // Get V for a specific head: output [cell_count, d_v]
    void get_v_head(int stream, int layer, int head, int d_v, std::vector<float> & out) const {
        const auto & ld = streams[stream].layers[layer];
        const int n_embd = ld.n_embd_v_gqa_computed();
        const int cc = ld.cell_count;
        out.resize(cc * d_v);
        for (int i = 0; i < cc; i++) {
            memcpy(out.data() + i * d_v,
                   ld.V.data() + i * n_embd + head * d_v,
                   d_v * sizeof(float));
        }
    }

private:
    template<typename T>
    static bool read_val(const uint8_t *& ptr, const uint8_t * end, T & val) {
        if (ptr + sizeof(T) > end) return false;
        memcpy(&val, ptr, sizeof(T));
        ptr += sizeof(T);
        return true;
    }

    static int type_size(int32_t type) {
        if (type == KV_COMPACT_GGML_TYPE_F32) return 4;
        if (type == KV_COMPACT_GGML_TYPE_F16) return 2;
        return 4; // fallback
    }

    // Convert raw data to float32 (supports F32 and F16)
    static void convert_to_f32(const uint8_t * src, int32_t type, float * dst, size_t n) {
        if (type == KV_COMPACT_GGML_TYPE_F32) {
            memcpy(dst, src, n * sizeof(float));
        } else if (type == KV_COMPACT_GGML_TYPE_F16) {
            const uint16_t * f16 = (const uint16_t *) src;
            for (size_t i = 0; i < n; i++) {
                // IEEE 754 half-precision to single-precision
                dst[i] = f16_to_f32(f16[i]);
            }
        } else {
            // Unsupported type — fill with zeros
            memset(dst, 0, n * sizeof(float));
        }
    }

    // Transpose V from [embd][cell] to [cell][embd] and convert to F32
    static void transpose_v_to_f32(const uint8_t * src, int32_t type, float * dst,
                                   uint32_t cell_count, uint32_t n_embd) {
        if (type == KV_COMPACT_GGML_TYPE_F32) {
            const float * f = (const float *) src;
            for (uint32_t d = 0; d < n_embd; d++) {
                for (uint32_t c = 0; c < cell_count; c++) {
                    dst[c * n_embd + d] = f[d * cell_count + c];
                }
            }
        } else if (type == KV_COMPACT_GGML_TYPE_F16) {
            const uint16_t * f16 = (const uint16_t *) src;
            for (uint32_t d = 0; d < n_embd; d++) {
                for (uint32_t c = 0; c < cell_count; c++) {
                    dst[c * n_embd + d] = f16_to_f32(f16[d * cell_count + c]);
                }
            }
        } else {
            memset(dst, 0, (size_t)cell_count * n_embd * sizeof(float));
        }
    }

    // Simple F16 → F32 conversion (IEEE 754)
    static float f16_to_f32(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) {
                // Zero
                uint32_t result = sign;
                float f;
                memcpy(&f, &result, 4);
                return f;
            }
            // Denormalized
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            exp = exp + (127 - 15);
        } else if (exp == 31) {
            // Inf / NaN
            uint32_t result = sign | 0x7F800000 | (mant << 13);
            float f;
            memcpy(&f, &result, 4);
            return f;
        } else {
            exp = exp + (127 - 15);
        }

        uint32_t result = sign | (exp << 23) | (mant << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }
};

// ============================================================================
// State buffer writer — builds compacted state from compaction results
// ============================================================================

// Convert float32 to F16 (IEEE 754)
static uint16_t f32_to_f16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);

    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; // underflow to zero
        mant = (mant | 0x400) >> (1 - exp);
        return (uint16_t)(sign | mant);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00); // overflow to inf
    }

    return (uint16_t)(sign | (exp << 10) | mant);
}

// Build a compacted state buffer from original parsed state + compaction results
//
// For each layer:
//   - K: copy original K rows for selected indices only
//   - V: write C_v (refitted values) for each head at selected positions
//
// selected_indices: [t] shared across all heads within a layer
// cv_per_head: [n_head_kv][t * d_v] refitted values per head
//
// Returns the new state buffer ready for llama_state_seq_set_data()
static std::vector<uint8_t> build_compacted_state(
        const parsed_kv_state & state,
        const std::vector<int> & selected_indices,
        // Per-layer, per-head C_v: cv_all[layer][head] = vector<float> of [t * d_v]
        const std::vector<std::vector<std::vector<float>>> & cv_all,
        int n_head_kv, int d_k, int d_v,
        uint32_t n_pos_per_embd = 1) {

    const int t = (int) selected_indices.size();

    // Estimate output size (generous overestimate)
    std::vector<uint8_t> out;
    out.reserve(state.raw.size()); // at most same size as original

    auto write = [&](const void * data, size_t sz) {
        const uint8_t * p = (const uint8_t *) data;
        out.insert(out.end(), p, p + sz);
    };

    // Write n_stream
    write(&state.n_stream, sizeof(state.n_stream));

    for (uint32_t s = 0; s < state.n_stream; s++) {
        const auto & sd = state.streams[s];

        if (sd.cell_count == 0) {
            uint32_t zero = 0;
            write(&zero, sizeof(zero));
            continue;
        }

        // Write compacted cell count
        uint32_t new_cell_count = (uint32_t) t;
        write(&new_cell_count, sizeof(new_cell_count));

        // Write cell metadata for selected indices only
        for (int j = 0; j < t; j++) {
            const auto & cell = sd.cells[selected_indices[j]];
            write(&cell.pos, sizeof(cell.pos));
            uint32_t n_seq_id = (uint32_t) cell.seq_ids.size();
            write(&n_seq_id, sizeof(n_seq_id));
            // Write IMROPE ext data (x,y positions) if applicable
            if (n_pos_per_embd > 1) {
                write(&cell.ext_x, sizeof(cell.ext_x));
                write(&cell.ext_y, sizeof(cell.ext_y));
            }
            for (const auto & sid : cell.seq_ids) {
                write(&sid, sizeof(sid));
            }
        }

        // Write v_trans and n_layer
        write(&sd.v_trans, sizeof(sd.v_trans));
        write(&sd.n_layer, sizeof(sd.n_layer));

        // Write K data per layer — original K rows at selected indices
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];

            // Write type and row size (same as original)
            write(&ld.k_type, sizeof(ld.k_type));
            write(&ld.k_size_row, sizeof(ld.k_size_row));

            const int n_embd_k = ld.n_embd_k_gqa();

            // Write selected K rows
            for (int j = 0; j < t; j++) {
                int orig_idx = selected_indices[j];
                const float * k_row = ld.K.data() + orig_idx * n_embd_k;

                if (ld.k_type == KV_COMPACT_GGML_TYPE_F32) {
                    write(k_row, n_embd_k * sizeof(float));
                } else if (ld.k_type == KV_COMPACT_GGML_TYPE_F16) {
                    std::vector<uint16_t> tmp(n_embd_k);
                    for (int d = 0; d < n_embd_k; d++) {
                        tmp[d] = f32_to_f16(k_row[d]);
                    }
                    write(tmp.data(), n_embd_k * sizeof(uint16_t));
                }
            }
        }

        // Write V data per layer — C_v (refitted values) at selected positions
        if (!sd.v_trans) {
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                const auto & ld = sd.layers[l];

                write(&ld.v_type, sizeof(ld.v_type));
                write(&ld.v_size_row, sizeof(ld.v_size_row));

                const int n_embd_v = ld.n_embd_v_gqa_computed();

                // Build full V rows from per-head C_v
                for (int j = 0; j < t; j++) {
                    std::vector<float> v_row(n_embd_v);
                    for (int h = 0; h < n_head_kv; h++) {
                        const float * cv = cv_all[l][h].data() + j * d_v;
                        memcpy(v_row.data() + h * d_v, cv, d_v * sizeof(float));
                    }

                    if (ld.v_type == KV_COMPACT_GGML_TYPE_F32) {
                        write(v_row.data(), n_embd_v * sizeof(float));
                    } else if (ld.v_type == KV_COMPACT_GGML_TYPE_F16) {
                        std::vector<uint16_t> tmp(n_embd_v);
                        for (int d = 0; d < n_embd_v; d++) {
                            tmp[d] = f32_to_f16(v_row[d]);
                        }
                        write(tmp.data(), n_embd_v * sizeof(uint16_t));
                    }
                }
            }
        } else {
            // Transposed V: write as [n_embd_v_gqa][t] per layer
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                const auto & ld = sd.layers[l];

                write(&ld.v_type, sizeof(ld.v_type));
                write(&ld.v_size_el, sizeof(ld.v_size_el));
                uint32_t n_embd_v = (uint32_t)(n_head_kv * d_v);
                write(&n_embd_v, sizeof(n_embd_v));

                // For each embedding dimension d, write t values (transposed)
                for (uint32_t d = 0; d < n_embd_v; d++) {
                    int h = d / d_v;
                    int di = d % d_v;
                    for (int j = 0; j < t; j++) {
                        float val = cv_all[l][h][j * d_v + di];
                        if (ld.v_type == KV_COMPACT_GGML_TYPE_F32) {
                            write(&val, sizeof(float));
                        } else if (ld.v_type == KV_COMPACT_GGML_TYPE_F16) {
                            uint16_t f16 = f32_to_f16(val);
                            write(&f16, sizeof(uint16_t));
                        }
                    }
                }
            }
        }
    }

    // Append trailing data (e.g., recurrent section for hybrid SSM+MoE models)
    if (!state.trailing_data.empty()) {
        write(state.trailing_data.data(), state.trailing_data.size());
    }

    return out;
}
