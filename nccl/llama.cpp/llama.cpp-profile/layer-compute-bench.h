#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

#include <unistd.h>
#include <sys/types.h>
#include <sys/resource.h>

#include "common.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"

extern "C" {
uint64_t layer_cpu_compute(struct ggml_cplan * cplan, struct ggml_tensor * node);
uint64_t layer_gpu_compute(ggml_tensor * src0_cpu, ggml_tensor * src1_cpu, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, void * context, void * data);
}

bool floatArraysEqual(const float* arr1, const float* arr2, size_t size, float epsilon);
bool parse_cpu_mask(const std::string & mask, bool(&boolmask)[GGML_MAX_N_THREADS]);
bool set_process_priority(enum ggml_sched_priority prio);
int32_t cpu_get_num_math();

struct Stats {
    uint64_t size;
    uint64_t n_params;
    std::vector<char> input_act;
    std::vector<char> output_act;
};

class ActCollector {
public:
    ActCollector() = default;
    void set_layers(std::vector<std::string> layers) { for (auto l: layers) m_stats.insert(std::make_pair(l, Stats{}));}
    struct Stats& get_layer(std::string layer_name) { return m_stats[layer_name]; }
    bool collect_activations(struct ggml_tensor * t, bool ask, void * user_data);
private:
    std::unordered_map<std::string, Stats> m_stats;
    std::mutex                             m_mutex;
};

static ActCollector g_collector;

// utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static bool tensor_buft_override_equal(const llama_model_tensor_buft_override& a, const llama_model_tensor_buft_override& b) {
    if (a.pattern != b.pattern) {
        // cString comparison that may be null
        if (a.pattern == nullptr || b.pattern == nullptr) {
            return false;
        }
        if (strcmp(a.pattern, b.pattern) != 0) {
            return false;
        }
    }
    if (a.buft != b.buft) {
        return false;
    }
    return true;
}

static bool vec_tensor_buft_override_equal(const std::vector<llama_model_tensor_buft_override>& a, const std::vector<llama_model_tensor_buft_override>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

static bool vec_vec_tensor_buft_override_equal(const std::vector<std::vector<llama_model_tensor_buft_override>>& a, const std::vector<std::vector<llama_model_tensor_buft_override>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!vec_tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

template <class T> static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template <typename T, typename F> static std::vector<std::string> transform_to_str(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template <typename T> static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T) v.size();
}

template <typename T> static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean   = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev  = std::sqrt(sq_sum / (T) (v.size() - 1) - mean * mean * (T) v.size() / (T) (v.size() - 1));
    return stdev;
}

static std::string get_cpu_info() {
    std::vector<std::string> cpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU || dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            cpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(cpu_list, ", ");
}

static std::string get_gpu_info() {
    std::vector<std::string> gpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(gpu_list, ", ");
}

// command line params
enum output_formats { NONE, CSV, JSON, JSONL, MARKDOWN, SQL };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case NONE:
            return "none";
        case CSV:
            return "csv";
        case JSON:
            return "json";
        case JSONL:
            return "jsonl";
        case MARKDOWN:
            return "md";
        case SQL:
            return "sql";
        default:
            GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "none") {
        format = NONE;
    } else if (s == "csv") {
        format = CSV;
    } else if (s == "json") {
        format = JSON;
    } else if (s == "jsonl") {
        format = JSONL;
    } else if (s == "md") {
        format = MARKDOWN;
    } else if (s == "sql") {
        format = SQL;
    } else {
        return false;
    }
    return true;
}

static const char * split_mode_str(llama_split_mode mode) {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE:
            return "none";
        case LLAMA_SPLIT_MODE_LAYER:
            return "layer";
        case LLAMA_SPLIT_MODE_ROW:
            return "row";
        default:
            GGML_ABORT("invalid split mode");
    }
}

static std::string pair_str(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

static std::vector<int> parse_int_range(const std::string & s) {
    // first[-last[(+|*)step]]
    std::regex range_regex(R"(^(\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))");

    std::smatch match;
    std::string::const_iterator search_start(s.cbegin());
    std::vector<int> result;
    while (std::regex_search(search_start, s.cend(), match, range_regex)) {
        int  first = std::stoi(match[1]);
        int  last  = match[2].matched ? std::stoi(match[2]) : first;
        char op    = match[3].matched ? match[3].str()[0] : '+';
        int  step  = match[4].matched ? std::stoi(match[4]) : 1;

        for (int i = first; i <= last;) {
            result.push_back(i);

            int prev_i = i;

            if (op == '+') {
                i += step;
            } else if (op == '*') {
                i *= step;
            } else {
                throw std::invalid_argument("invalid range format");
            }

            if (i <= prev_i) {
                throw std::invalid_argument("invalid range");
            }
        }
        search_start = match.suffix().first;
    }

    if (search_start != s.cend()) {
        throw std::invalid_argument("invalid range format");
    }

    return result;
}

struct cmd_params {
    std::vector<std::string>         model;
    std::vector<std::string>         layer;
    std::vector<int>                 n_prompt;
    std::vector<int>                 n_gen;
    std::vector<std::pair<int, int>> n_pg;
    std::vector<int>                 n_depth;
    std::vector<int>                 n_batch;
    std::vector<int>                 n_ubatch;
    std::vector<ggml_type>           type_k;
    std::vector<ggml_type>           type_v;
    std::vector<float>               defrag_thold;
    std::vector<int>                 n_threads;
    std::vector<std::string>         cpu_mask;
    std::vector<bool>                cpu_strict;
    std::vector<int>                 poll;
    std::vector<int>                 n_gpu_layers;
    std::vector<std::string>         rpc_servers;
    std::vector<llama_split_mode>    split_mode;
    std::vector<int>                 main_gpu;
    std::vector<bool>                no_kv_offload;
    std::vector<bool>                flash_attn;
    std::vector<std::vector<float>>  tensor_split;
    std::vector<std::vector<llama_model_tensor_buft_override>> tensor_buft_overrides;
    std::vector<bool>                use_mmap;
    std::vector<bool>                embeddings;
    std::vector<bool>                no_op_offload;
    ggml_numa_strategy               numa;
    int                              reps;
    ggml_sched_priority              prio;
    int                              delay;
    bool                             verbose;
    bool                             progress;
    bool                             no_warmup;
    output_formats                   output_format;
    output_formats                   output_format_stderr;
};

static const cmd_params cmd_params_defaults = {
    /* model                */ { "models/7B/ggml-model-q4_0.gguf" },
    /* layer                */ { "blk.0.attn_q.weight" },
    /* n_prompt             */ { 512 },
    /* n_gen                */ { 128 },
    /* n_pg                 */ {},
    /* n_depth              */ { 0 },
    /* n_batch              */ { 2048 },
    /* n_ubatch             */ { 512 },
    /* type_k               */ { GGML_TYPE_F16 },
    /* type_v               */ { GGML_TYPE_F16 },
    /* defrag_thold         */ { -1.0f },
    /* n_threads            */ { cpu_get_num_math() },
    /* cpu_mask             */ { "0x0" },
    /* cpu_strict           */ { false },
    /* poll                 */ { 50 },
    /* n_gpu_layers         */ { 99 },
    /* rpc_servers          */ { "" },
    /* split_mode           */ { LLAMA_SPLIT_MODE_LAYER },
    /* main_gpu             */ { 0 },
    /* no_kv_offload        */ { false },
    /* flash_attn           */ { false },
    /* tensor_split         */ { std::vector<float>(llama_max_devices(), 0.0f) },
    /* tensor_buft_overrides*/ { std::vector<llama_model_tensor_buft_override>{ { nullptr, nullptr } } },
    /* use_mmap             */ { true },
    /* embeddings           */ { false },
    /* no_op_offload        */ { false },
    /* numa                 */ GGML_NUMA_STRATEGY_DISABLED,
    /* reps                 */ 5,
    /* prio                 */ GGML_SCHED_PRIO_NORMAL,
    /* delay                */ 0,
    /* verbose              */ false,
    /* progress             */ false,
    /* no_warmup            */ false,
    /* output_format        */ MARKDOWN,
    /* output_format_stderr */ NONE,
};

static void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  --numa <distribute|isolate|numactl>       numa mode (default: disabled)\n");
    printf("  -r, --repetitions <n>                     number of times to repeat each test (default: %d)\n",
           cmd_params_defaults.reps);
    printf("  --prio <-1|0|1|2|3>                          process/thread priority (default: %d)\n",
           cmd_params_defaults.prio);
    printf("  --delay <0...N> (seconds)                 delay between each test (default: %d)\n",
           cmd_params_defaults.delay);
    printf("  -o, --output <csv|json|jsonl|md|sql>      output format printed to stdout (default: %s)\n",
           output_format_str(cmd_params_defaults.output_format));
    printf("  -oe, --output-err <csv|json|jsonl|md|sql> output format printed to stderr (default: %s)\n",
           output_format_str(cmd_params_defaults.output_format_stderr));
    printf("  -v, --verbose                             verbose output\n");
    printf("  --progress                                print test progress indicators\n");
    printf("  --no-warmup                               skip warmup runs before benchmarking\n");
    printf("\n");
    printf("test parameters:\n");
    printf("  -m, --model <filename>                    (default: %s)\n", join(cmd_params_defaults.model, ",").c_str());
    printf("  -l, --layer <layername>                   (default: %s)\n", join(cmd_params_defaults.layer, ",").c_str());
    printf("  -p, --n-prompt <n>                        (default: %s)\n",
           join(cmd_params_defaults.n_prompt, ",").c_str());
    printf("  -n, --n-gen <n>                           (default: %s)\n", join(cmd_params_defaults.n_gen, ",").c_str());
    printf("  -pg <pp,tg>                               (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.n_pg, pair_str), ",").c_str());
    printf("  -d, --n-depth <n>                         (default: %s)\n",
           join(cmd_params_defaults.n_depth, ",").c_str());
    printf("  -b, --batch-size <n>                      (default: %s)\n",
           join(cmd_params_defaults.n_batch, ",").c_str());
    printf("  -ub, --ubatch-size <n>                    (default: %s)\n",
           join(cmd_params_defaults.n_ubatch, ",").c_str());
    printf("  -ctk, --cache-type-k <t>                  (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.type_k, ggml_type_name), ",").c_str());
    printf("  -ctv, --cache-type-v <t>                  (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.type_v, ggml_type_name), ",").c_str());
    printf("  -dt, --defrag-thold <f>                   (default: %s)\n",
           join(cmd_params_defaults.defrag_thold, ",").c_str());
    printf("  -t, --threads <n>                         (default: %s)\n",
           join(cmd_params_defaults.n_threads, ",").c_str());
    printf("  -C, --cpu-mask <hex,hex>                  (default: %s)\n",
           join(cmd_params_defaults.cpu_mask, ",").c_str());
    printf("  --cpu-strict <0|1>                        (default: %s)\n",
           join(cmd_params_defaults.cpu_strict, ",").c_str());
    printf("  --poll <0...100>                          (default: %s)\n", join(cmd_params_defaults.poll, ",").c_str());
    printf("  -ngl, --n-gpu-layers <n>                  (default: %s)\n",
           join(cmd_params_defaults.n_gpu_layers, ",").c_str());
    if (llama_supports_rpc()) {
        printf("  -rpc, --rpc <rpc_servers>                 (default: %s)\n",
               join(cmd_params_defaults.rpc_servers, ",").c_str());
    }
    printf("  -sm, --split-mode <none|layer|row>        (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.split_mode, split_mode_str), ",").c_str());
    printf("  -mg, --main-gpu <i>                       (default: %s)\n",
           join(cmd_params_defaults.main_gpu, ",").c_str());
    printf("  -nkvo, --no-kv-offload <0|1>              (default: %s)\n",
           join(cmd_params_defaults.no_kv_offload, ",").c_str());
    printf("  -fa, --flash-attn <0|1>                   (default: %s)\n",
           join(cmd_params_defaults.flash_attn, ",").c_str());
    printf("  -mmp, --mmap <0|1>                        (default: %s)\n",
           join(cmd_params_defaults.use_mmap, ",").c_str());
    printf("  -embd, --embeddings <0|1>                 (default: %s)\n",
           join(cmd_params_defaults.embeddings, ",").c_str());
    printf("  -ts, --tensor-split <ts0/ts1/..>          (default: 0)\n");
    printf("  -ot --override-tensors <tensor name pattern>=<buffer type>;...\n");
    printf("                                            (default: disabled)\n");
    printf("  -nopo, --no-op-offload <0|1>              (default: 0)\n");
    printf("\n");
    printf(
        "Multiple values can be given for each parameter by separating them with ','\n"
        "or by specifying the parameter multiple times. Ranges can be given as\n"
        "'first-last' or 'first-last+step' or 'first-last*mult'.\n");
}

static ggml_type ggml_type_from_name(const std::string & s) {
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "bf16") {
        return GGML_TYPE_BF16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }

    return GGML_TYPE_COUNT;
}

static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params        params;
    std::string       arg;
    bool              invalid_param = false;
    const std::string arg_prefix    = "--";
    const char        split_delim   = ',';

    params.verbose              = cmd_params_defaults.verbose;
    params.output_format        = cmd_params_defaults.output_format;
    params.output_format_stderr = cmd_params_defaults.output_format_stderr;
    params.reps                 = cmd_params_defaults.reps;
    params.numa                 = cmd_params_defaults.numa;
    params.prio                 = cmd_params_defaults.prio;
    params.delay                = cmd_params_defaults.delay;
    params.progress             = cmd_params_defaults.progress;
    params.no_warmup            = cmd_params_defaults.no_warmup;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        try {
            if (arg == "-h" || arg == "--help") {
                print_usage(argc, argv);
                exit(0);
            } else if (arg == "-m" || arg == "--model") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.model.insert(params.model.end(), p.begin(), p.end());
            } else if (arg == "-l" || arg == "--layer") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.layer.insert(params.layer.end(), p.begin(), p.end());
            } else if (arg == "-p" || arg == "--n-prompt") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_prompt.insert(params.n_prompt.end(), p.begin(), p.end());
            } else if (arg == "-n" || arg == "--n-gen") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_gen.insert(params.n_gen.end(), p.begin(), p.end());
            } else if (arg == "-pg") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], ',');
                if (p.size() != 2) {
                    invalid_param = true;
                    break;
                }
                params.n_pg.push_back({ std::stoi(p[0]), std::stoi(p[1]) });
            } else if (arg == "-d" || arg == "--n-depth") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_depth.insert(params.n_depth.end(), p.begin(), p.end());
            } else if (arg == "-b" || arg == "--batch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_batch.insert(params.n_batch.end(), p.begin(), p.end());
            } else if (arg == "-ub" || arg == "--ubatch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_ubatch.insert(params.n_ubatch.end(), p.begin(), p.end());
            } else if (arg == "-ctk" || arg == "--cache-type-k") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<ggml_type> types;
                for (const auto & t : p) {
                    ggml_type gt = ggml_type_from_name(t);
                    if (gt == GGML_TYPE_COUNT) {
                        invalid_param = true;
                        break;
                    }
                    types.push_back(gt);
                }
                if (invalid_param) {
                    break;
                }
                params.type_k.insert(params.type_k.end(), types.begin(), types.end());
            } else if (arg == "-ctv" || arg == "--cache-type-v") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<ggml_type> types;
                for (const auto & t : p) {
                    ggml_type gt = ggml_type_from_name(t);
                    if (gt == GGML_TYPE_COUNT) {
                        invalid_param = true;
                        break;
                    }
                    types.push_back(gt);
                }
                if (invalid_param) {
                    break;
                }
                params.type_v.insert(params.type_v.end(), types.begin(), types.end());
            } else if (arg == "-dt" || arg == "--defrag-thold") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<float>(argv[i], split_delim);
                params.defrag_thold.insert(params.defrag_thold.end(), p.begin(), p.end());
            } else if (arg == "-t" || arg == "--threads") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_threads.insert(params.n_threads.end(), p.begin(), p.end());
            } else if (arg == "-C" || arg == "--cpu-mask") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.cpu_mask.insert(params.cpu_mask.end(), p.begin(), p.end());
            } else if (arg == "--cpu-strict") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.cpu_strict.insert(params.cpu_strict.end(), p.begin(), p.end());
            } else if (arg == "--poll") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.poll.insert(params.poll.end(), p.begin(), p.end());
            } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_gpu_layers.insert(params.n_gpu_layers.end(), p.begin(), p.end());
            } else if (llama_supports_rpc() && (arg == "-rpc" || arg == "--rpc")) {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.rpc_servers.push_back(argv[i]);
            } else if (arg == "-sm" || arg == "--split-mode") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<llama_split_mode> modes;
                for (const auto & m : p) {
                    llama_split_mode mode;
                    if (m == "none") {
                        mode = LLAMA_SPLIT_MODE_NONE;
                    } else if (m == "layer") {
                        mode = LLAMA_SPLIT_MODE_LAYER;
                    } else if (m == "row") {
                        mode = LLAMA_SPLIT_MODE_ROW;
                    } else {
                        invalid_param = true;
                        break;
                    }
                    modes.push_back(mode);
                }
                if (invalid_param) {
                    break;
                }
                params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());
            } else if (arg == "-mg" || arg == "--main-gpu") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.main_gpu = parse_int_range(argv[i]);
            } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.no_kv_offload.insert(params.no_kv_offload.end(), p.begin(), p.end());
            } else if (arg == "--numa") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                std::string value(argv[i]);
                if (value == "distribute" || value == "") {
                    params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
                } else if (value == "isolate") {
                    params.numa = GGML_NUMA_STRATEGY_ISOLATE;
                } else if (value == "numactl") {
                    params.numa = GGML_NUMA_STRATEGY_NUMACTL;
                } else {
                    invalid_param = true;
                    break;
                }
            } else if (arg == "-fa" || arg == "--flash-attn") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.flash_attn.insert(params.flash_attn.end(), p.begin(), p.end());
            } else if (arg == "-mmp" || arg == "--mmap") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.use_mmap.insert(params.use_mmap.end(), p.begin(), p.end());
            } else if (arg == "-embd" || arg == "--embeddings") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.embeddings.insert(params.embeddings.end(), p.begin(), p.end());
            } else if (arg == "-nopo" || arg == "--no-op-offload") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.no_op_offload.insert(params.no_op_offload.end(), p.begin(), p.end());
            } else if (arg == "-ts" || arg == "--tensor-split") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                for (auto ts : string_split<std::string>(argv[i], split_delim)) {
                    // split string by ; and /
                    const std::regex           regex{ R"([;/]+)" };
                    std::sregex_token_iterator it{ ts.begin(), ts.end(), regex, -1 };
                    std::vector<std::string>   split_arg{ it, {} };
                    GGML_ASSERT(split_arg.size() <= llama_max_devices());

                    std::vector<float> tensor_split(llama_max_devices());
                    for (size_t i = 0; i < llama_max_devices(); ++i) {
                        if (i < split_arg.size()) {
                            tensor_split[i] = std::stof(split_arg[i]);
                        } else {
                            tensor_split[i] = 0.0f;
                        }
                    }
                    params.tensor_split.push_back(tensor_split);
                }
            } else if (arg == "-ot" || arg == "--override-tensor") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto * value = argv[i];
                /* static */ std::map<std::string, ggml_backend_buffer_type_t> buft_list;
                if (buft_list.empty()) {
                    // enumerate all the devices and add their buffer types to the list
                    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                        auto * dev = ggml_backend_dev_get(i);
                        auto * buft = ggml_backend_dev_buffer_type(dev);
                        if (buft) {
                            buft_list[ggml_backend_buft_name(buft)] = buft;
                        }
                    }
                }
                auto override_group_span_len = std::strcspn(value, ",");
                bool last_group = false;
                do {
                    if (override_group_span_len == 0) {
                        // Adds an empty override-tensors for an empty span
                        params.tensor_buft_overrides.push_back({{}});
                        if (value[override_group_span_len] == '\0') {
                            value = &value[override_group_span_len];
                            last_group = true;
                        } else {
                            value = &value[override_group_span_len + 1];
                            override_group_span_len = std::strcspn(value, ",");
                        }
                        continue;
                    }
                    // Stamps null terminators into the argv
                    // value for this option to avoid the
                    // memory leak present in the implementation
                    // over in arg.cpp. Acceptable because we
                    // only parse these args once in this program.
                    auto * override_group = value;
                    if (value[override_group_span_len] == '\0') {
                        value = &value[override_group_span_len];
                        last_group = true;
                    } else {
                        value[override_group_span_len] = '\0';
                        value = &value[override_group_span_len + 1];
                    }
                    std::vector<llama_model_tensor_buft_override> group_tensor_buft_overrides{};
                    auto override_span_len = std::strcspn(override_group, ";");
                    while (override_span_len > 0) {
                        auto * override = override_group;
                        if (override_group[override_span_len] != '\0') {
                            override_group[override_span_len] = '\0';
                            override_group = &override_group[override_span_len + 1];
                        } else {
                            override_group = &override_group[override_span_len];
                        }
                        auto tensor_name_span_len = std::strcspn(override, "=");
                        if (tensor_name_span_len >= override_span_len) {
                            invalid_param = true;
                            break;
                        }
                        override[tensor_name_span_len] = '\0';
                        auto * tensor_name = override;
                        auto * buffer_type = &override[tensor_name_span_len + 1];
                        if (buft_list.find(buffer_type) == buft_list.end()) {
                            printf("error: unrecognized buffer type '%s'\n", buffer_type);
                            printf("Available buffer types:\n");
                            for (const auto & it : buft_list) {
                                printf("  %s\n", ggml_backend_buft_name(it.second));
                            }
                            invalid_param = true;
                            break;
                        }
                        group_tensor_buft_overrides.push_back({tensor_name, buft_list.at(buffer_type)});
                        override_span_len = std::strcspn(override_group, ";");
                    }
                    if (invalid_param) {
                        break;
                    }
                    group_tensor_buft_overrides.push_back({nullptr,nullptr});
                    params.tensor_buft_overrides.push_back(group_tensor_buft_overrides);
                    override_group_span_len = std::strcspn(value, ",");
                } while (!last_group);
            } else if (arg == "-r" || arg == "--repetitions") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.reps = std::stoi(argv[i]);
            } else if (arg == "--prio") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.prio = (enum ggml_sched_priority) std::stoi(argv[i]);
            } else if (arg == "--delay") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.delay = std::stoi(argv[i]);
            } else if (arg == "-o" || arg == "--output") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                invalid_param = !output_format_from_str(argv[i], params.output_format);
            } else if (arg == "-oe" || arg == "--output-err") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                invalid_param = !output_format_from_str(argv[i], params.output_format_stderr);
            } else if (arg == "-v" || arg == "--verbose") {
                params.verbose = true;
            } else if (arg == "--progress") {
                params.progress = true;
            } else if (arg == "--no-warmup") {
                params.no_warmup = true;
            } else {
                invalid_param = true;
                break;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            invalid_param = true;
            break;
        }
    }

    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }

    // set defaults
    if (params.model.empty()) {
        params.model = cmd_params_defaults.model;
    }
    if (params.n_prompt.empty()) {
        params.n_prompt = cmd_params_defaults.n_prompt;
    }
    if (params.n_gen.empty()) {
        params.n_gen = cmd_params_defaults.n_gen;
    }
    if (params.n_pg.empty()) {
        params.n_pg = cmd_params_defaults.n_pg;
    }
    if (params.n_depth.empty()) {
        params.n_depth = cmd_params_defaults.n_depth;
    }
    if (params.n_batch.empty()) {
        params.n_batch = cmd_params_defaults.n_batch;
    }
    if (params.n_ubatch.empty()) {
        params.n_ubatch = cmd_params_defaults.n_ubatch;
    }
    if (params.type_k.empty()) {
        params.type_k = cmd_params_defaults.type_k;
    }
    if (params.type_v.empty()) {
        params.type_v = cmd_params_defaults.type_v;
    }
    if (params.defrag_thold.empty()) {
        params.defrag_thold = cmd_params_defaults.defrag_thold;
    }
    if (params.n_gpu_layers.empty()) {
        params.n_gpu_layers = cmd_params_defaults.n_gpu_layers;
    }
    if (params.rpc_servers.empty()) {
        params.rpc_servers = cmd_params_defaults.rpc_servers;
    }
    if (params.split_mode.empty()) {
        params.split_mode = cmd_params_defaults.split_mode;
    }
    if (params.main_gpu.empty()) {
        params.main_gpu = cmd_params_defaults.main_gpu;
    }
    if (params.no_kv_offload.empty()) {
        params.no_kv_offload = cmd_params_defaults.no_kv_offload;
    }
    if (params.flash_attn.empty()) {
        params.flash_attn = cmd_params_defaults.flash_attn;
    }
    if (params.tensor_split.empty()) {
        params.tensor_split = cmd_params_defaults.tensor_split;
    }
    if (params.tensor_buft_overrides.empty()) {
        params.tensor_buft_overrides = cmd_params_defaults.tensor_buft_overrides;
    }
    if (params.use_mmap.empty()) {
        params.use_mmap = cmd_params_defaults.use_mmap;
    }
    if (params.embeddings.empty()) {
        params.embeddings = cmd_params_defaults.embeddings;
    }
    if (params.no_op_offload.empty()) {
        params.no_op_offload = cmd_params_defaults.no_op_offload;
    }
    if (params.n_threads.empty()) {
        params.n_threads = cmd_params_defaults.n_threads;
    }
    if (params.cpu_mask.empty()) {
        params.cpu_mask = cmd_params_defaults.cpu_mask;
    }
    if (params.cpu_strict.empty()) {
        params.cpu_strict = cmd_params_defaults.cpu_strict;
    }
    if (params.poll.empty()) {
        params.poll = cmd_params_defaults.poll;
    }

    return params;
}

struct cmd_params_instance {
    std::string        model;
    std::vector<std::string> layers;
    int                n_prompt;
    int                n_gen;
    int                n_depth;
    int                n_batch;
    int                n_ubatch;
    ggml_type          type_k;
    ggml_type          type_v;
    float              defrag_thold;
    int                n_threads;
    std::string        cpu_mask;
    bool               cpu_strict;
    int                poll;
    int                n_gpu_layers;
    std::string        rpc_servers_str;
    llama_split_mode   split_mode;
    int                main_gpu;
    bool               no_kv_offload;
    bool               flash_attn;
    std::vector<float> tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    bool               use_mmap;
    bool               embeddings;
    bool               no_op_offload;
    ggml_backend_sched_eval_callback cb_eval;

    llama_model_params to_llama_mparams() const {
        llama_model_params mparams = llama_model_default_params();

        mparams.n_gpu_layers = n_gpu_layers;
        mparams.split_mode   = split_mode;
        mparams.main_gpu     = main_gpu;
        mparams.tensor_split = tensor_split.data();
        mparams.use_mmap     = use_mmap;

        if (tensor_buft_overrides.empty()) {
            mparams.tensor_buft_overrides = nullptr;
        } else {
            GGML_ASSERT(tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
            mparams.tensor_buft_overrides = tensor_buft_overrides.data();
        }

        return mparams;
    }

    bool equal_mparams(const cmd_params_instance & other) const {
        return model == other.model && n_gpu_layers == other.n_gpu_layers && rpc_servers_str == other.rpc_servers_str &&
               split_mode == other.split_mode && main_gpu == other.main_gpu && use_mmap == other.use_mmap &&
               tensor_split == other.tensor_split && vec_tensor_buft_override_equal(tensor_buft_overrides, other.tensor_buft_overrides);
    }

    llama_context_params to_llama_cparams() const {
        llama_context_params cparams = llama_context_default_params();

        cparams.n_ctx        = n_prompt + n_gen + n_depth;
        cparams.n_batch      = n_batch;
        cparams.n_ubatch     = n_ubatch;
        cparams.type_k       = type_k;
        cparams.type_v       = type_v;
        cparams.defrag_thold = defrag_thold;
        cparams.offload_kqv  = !no_kv_offload;
        cparams.flash_attn   = flash_attn;
        cparams.embeddings   = embeddings;
        cparams.op_offload   = !no_op_offload;
        cparams.swa_full     = false;
        cparams.cb_eval      = cb_eval;

        return cparams;
    }
};

static std::vector<cmd_params_instance> get_cmd_params_instances(const cmd_params & params) {
    std::vector<cmd_params_instance> instances;

    // this ordering minimizes the number of times that each model needs to be reloaded
    // clang-format off
    for (const auto & m : params.model)
    for (const auto & nl : params.n_gpu_layers)
    for (const auto & rpc : params.rpc_servers)
    for (const auto & sm : params.split_mode)
    for (const auto & mg : params.main_gpu)
    for (const auto & ts : params.tensor_split)
    for (const auto & ot : params.tensor_buft_overrides)
    for (const auto & mmp : params.use_mmap)
    for (const auto & embd : params.embeddings)
    for (const auto & nopo : params.no_op_offload)
    for (const auto & nb : params.n_batch)
    for (const auto & nub : params.n_ubatch)
    for (const auto & tk : params.type_k)
    for (const auto & tv : params.type_v)
    for (const auto & defrag_thold : params.defrag_thold)
    for (const auto & nkvo : params.no_kv_offload)
    for (const auto & fa : params.flash_attn)
    for (const auto & nt : params.n_threads)
    for (const auto & cm : params.cpu_mask)
    for (const auto & cs : params.cpu_strict)
    for (const auto & nd : params.n_depth)
    for (const auto & pl : params.poll) {
        for (const auto & n_prompt : params.n_prompt) {
            if (n_prompt == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .layers       = */ params.layer,
                /* .n_prompt     = */ n_prompt,
                /* .n_gen        = */ 0,
                /* .n_depth      = */ nd,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .defrag_thold = */ defrag_thold,
                /* .n_threads    = */ nt,
                /* .cpu_mask     = */ cm,
                /* .cpu_strict   = */ cs,
                /* .poll         = */ pl,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .no_op_offload= */ nopo,
                /* .cb_eval= */       nullptr
            };
            instances.push_back(instance);
        }

        for (const auto & n_gen : params.n_gen) {
            if (n_gen == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .layers       = */ params.layer,
                /* .n_prompt     = */ 0,
                /* .n_gen        = */ n_gen,
                /* .n_depth      = */ nd,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .defrag_thold = */ defrag_thold,
                /* .n_threads    = */ nt,
                /* .cpu_mask     = */ cm,
                /* .cpu_strict   = */ cs,
                /* .poll         = */ pl,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .no_op_offload= */ nopo,
                /* .cb_eval= */       nullptr
            };
            instances.push_back(instance);
        }

        for (const auto & n_pg : params.n_pg) {
            if (n_pg.first == 0 && n_pg.second == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .layers       = */ params.layer,
                /* .n_prompt     = */ n_pg.first,
                /* .n_gen        = */ n_pg.second,
                /* .n_depth      = */ nd,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .defrag_thold = */ defrag_thold,
                /* .n_threads    = */ nt,
                /* .cpu_mask     = */ cm,
                /* .cpu_strict   = */ cs,
                /* .poll         = */ pl,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .no_op_offload= */ nopo,
                /* .cb_eval= */       nullptr
            };
            instances.push_back(instance);
        }
    }
    // clang-format on

    return instances;
}

struct layer_tinfo {
    std::string name;
    uint64_t samples_ns;
    double ts;
};

struct test {
    const std::string        cpu_info;
    const std::string        gpu_info;
    std::string              model_filename;
    std::vector<struct layer_tinfo> layers;
    std::string              model_type;
    uint64_t                 model_size;
    uint64_t                 model_n_params;
    int                      n_batch;
    int                      n_ubatch;
    int                      n_threads;
    std::string              cpu_mask;
    bool                     cpu_strict;
    int                      poll;
    ggml_type                type_k;
    ggml_type                type_v;
    float                    defrag_thold;
    int                      n_gpu_layers;
    llama_split_mode         split_mode;
    int                      main_gpu;
    bool                     no_kv_offload;
    bool                     flash_attn;
    std::vector<float>       tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    bool                     use_mmap;
    bool                     embeddings;
    bool                     no_op_offload;
    int                      n_prompt;
    int                      n_gen;
    int                      n_depth;
    std::string              test_time;
    uint64_t    samples_ns;

    test(const cmd_params_instance & inst, const llama_model * lmodel, const llama_context * ctx) :
        cpu_info(get_cpu_info()),
        gpu_info(get_gpu_info()) {

        model_filename = inst.model;
        char buf[128];
        llama_model_desc(lmodel, buf, sizeof(buf));
        model_type     = buf;
        model_size     = llama_model_size(lmodel);
        model_n_params = llama_model_n_params(lmodel);
        n_batch        = inst.n_batch;
        n_ubatch       = inst.n_ubatch;
        n_threads      = inst.n_threads;
        cpu_mask       = inst.cpu_mask;
        cpu_strict     = inst.cpu_strict;
        poll           = inst.poll;
        type_k         = inst.type_k;
        type_v         = inst.type_v;
        defrag_thold   = inst.defrag_thold;
        n_gpu_layers   = inst.n_gpu_layers;
        split_mode     = inst.split_mode;
        main_gpu       = inst.main_gpu;
        no_kv_offload  = inst.no_kv_offload;
        flash_attn     = inst.flash_attn;
        tensor_split   = inst.tensor_split;
        tensor_buft_overrides = inst.tensor_buft_overrides;
        use_mmap       = inst.use_mmap;
        embeddings     = inst.embeddings;
        no_op_offload  = inst.no_op_offload;
        n_prompt       = inst.n_prompt;
        n_gen          = inst.n_gen;
        n_depth        = inst.n_depth;
        // RFC 3339 date-time format
        time_t t       = time(NULL);
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;
        samples_ns = 0;
        for (const auto& layer_name: inst.layers) {
            layers.push_back(layer_tinfo{
                /*.name = */layer_name,
                /*.samples_ns = */0,
                /*.ts = */0.0
            });
        }

        (void) ctx;
    }

    uint64_t model_ns() const { return samples_ns; }

    uint64_t stdev_ns() const { return 0; }

    std::vector<double> layer_ts() const {
        int n_tokens = n_prompt + n_gen;
        std::vector<double> ts;
        for (const auto& layer: layers) {
            ts.push_back(1e9 * n_tokens / layer.samples_ns);
        }
        return ts;
    }

    double model_ts() const {
        int n_tokens = n_prompt + n_gen;
        return 1e9 * n_tokens / model_ns();
    }

    double stdev_ts() const { return ::stdev(layer_ts()); }

    static std::string get_backend() {
        std::vector<std::string> backends;
        for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
            auto *      reg  = ggml_backend_reg_get(i);
            std::string name = ggml_backend_reg_name(reg);
            if (name != "CPU") {
                backends.push_back(ggml_backend_reg_name(reg));
            }
        }
        return backends.empty() ? "CPU" : join(backends, ",");
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "cpu_info",       "gpu_info",   "backends",     "model_filename",
            "model_type",   "model_size",   "model_n_params", "n_batch",    "n_ubatch",     "n_threads",
            "cpu_mask",     "cpu_strict",   "poll",           "type_k",     "type_v",       "n_gpu_layers",
            "split_mode",   "main_gpu",     "no_kv_offload",  "flash_attn", "tensor_split", "tensor_buft_overrides",
            "defrag_thold",
            "use_mmap",     "embeddings",   "no_op_offload",   "n_prompt",       "n_gen",      "n_depth",      "test_time",
            "model_ns",       "stddev_ns",    "model_ts",         "stddev_ts",
        };
        return fields;
    }

    enum field_type { STRING, BOOL, INT, FLOAT };

    static field_type get_field_type(const std::string & field) {
        if (field == "n_batch" || field == "n_ubatch" || field == "n_threads" ||
            field == "poll" || field == "model_size" || field == "model_n_params" || field == "n_gpu_layers" ||
            field == "main_gpu" || field == "n_prompt" || field == "n_gen" || field == "n_depth" ||
            field == "model_ns" || field == "stddev_ns" || field == "no_op_offload") {
            return INT;
        }
        if (field == "f16_kv" || field == "no_kv_offload" || field == "cpu_strict" || field == "flash_attn" ||
            field == "use_mmap" || field == "embeddings") {
            return BOOL;
        }
        if (field == "model_ts" || field == "stddev_ts" || field == "defrag_thold") {
            return FLOAT;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        std::string tensor_split_str;
        std::string tensor_buft_overrides_str;
        int         max_nonzero = 0;
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (tensor_split[i] > 0) {
                max_nonzero = i;
            }
        }
        for (int i = 0; i <= max_nonzero; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f", tensor_split[i]);
            tensor_split_str += buf;
            if (i < max_nonzero) {
                tensor_split_str += "/";
            }
        }
        if (tensor_buft_overrides.size() == 1) {
            // Last element of tensor_buft_overrides is always a null pattern
            // so if it is only one element long, it must be a null pattern.
            GGML_ASSERT(tensor_buft_overrides[0].pattern == nullptr);
            tensor_buft_overrides_str += "none";
        } else {
            for (size_t i = 0; i < tensor_buft_overrides.size()-1; i++) {
                // Last element of tensor_buft_overrides is always a null pattern
                if (tensor_buft_overrides[i].pattern == nullptr) {
                    tensor_buft_overrides_str += "none";
                } else {
                    tensor_buft_overrides_str += tensor_buft_overrides[i].pattern;
                    tensor_buft_overrides_str += "=";
                    tensor_buft_overrides_str += ggml_backend_buft_name(tensor_buft_overrides[i].buft);
                }
                if (i + 2 < tensor_buft_overrides.size()) {
                    tensor_buft_overrides_str += ";";
                }
            }
        }
        std::vector<std::string> values = {cpu_info,
                                            gpu_info,
                                            get_backend(),
                                            model_filename,
                                            model_type,
                                            std::to_string(model_size),
                                            std::to_string(model_n_params),
                                            std::to_string(n_batch),
                                            std::to_string(n_ubatch),
                                            std::to_string(n_threads),
                                            cpu_mask,
                                            std::to_string(cpu_strict),
                                            std::to_string(poll),
                                            ggml_type_name(type_k),
                                            ggml_type_name(type_v),
                                            std::to_string(n_gpu_layers),
                                            split_mode_str(split_mode),
                                            std::to_string(main_gpu),
                                            std::to_string(no_kv_offload),
                                            std::to_string(flash_attn),
                                            tensor_split_str,
                                            tensor_buft_overrides_str,
                                            std::to_string(defrag_thold),
                                            std::to_string(use_mmap),
                                            std::to_string(embeddings),
                                            std::to_string(no_op_offload),
                                            std::to_string(n_prompt),
                                            std::to_string(n_gen),
                                            std::to_string(n_depth),
                                            test_time,
                                            std::to_string(model_ns()),
                                            std::to_string(stdev_ns()),
                                            std::to_string(model_ts()),
                                            std::to_string(stdev_ts()) };
        return values;
    }

    std::map<std::string, std::string> get_map() const {
        std::map<std::string, std::string> map;
        auto                               fields = get_fields();
        auto                               values = get_values();
        std::transform(fields.begin(), fields.end(), values.begin(), std::inserter(map, map.end()),
                       std::make_pair<const std::string &, const std::string &>);
        return map;
    }
};

struct printer {
    virtual ~printer() {}

    FILE * fout;

    virtual void print_header(const cmd_params & params) { (void) params; }

    virtual void print_test(const test & t) = 0;
};

struct markdown_printer : public printer {
    std::vector<std::string> fields;

    static int get_field_width(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "t/s") {
            return 20;
        }
        if (field == "size" || field == "params") {
            return 12;
        }
        if (field == "n_gpu_layers") {
            return 3;
        }
        if (field == "n_threads") {
            return 7;
        }
        if (field == "n_batch") {
            return 7;
        }
        if (field == "n_ubatch") {
            return 8;
        }
        if (field == "type_k" || field == "type_v") {
            return 6;
        }
        if (field == "split_mode") {
            return 5;
        }
        if (field == "flash_attn") {
            return 2;
        }
        if (field == "use_mmap") {
            return 4;
        }
        if (field == "test") {
            return 15;
        }
        if (field == "no_op_offload") {
            return 4;
        }

        int width = std::max((int) field.length(), 10);

        if (test::get_field_type(field) == test::STRING) {
            return -width;
        }
        return width;
    }

    static std::string get_field_display_name(const std::string & field) {
        if (field == "n_gpu_layers") {
            return "ngl";
        }
        if (field == "split_mode") {
            return "sm";
        }
        if (field == "n_threads") {
            return "threads";
        }
        if (field == "no_kv_offload") {
            return "nkvo";
        }
        if (field == "flash_attn") {
            return "fa";
        }
        if (field == "use_mmap") {
            return "mmap";
        }
        if (field == "embeddings") {
            return "embd";
        }
        if (field == "no_op_offload") {
            return "nopo";
        }
        if (field == "tensor_split") {
            return "ts";
        }
        if (field == "tensor_buft_overrides") {
            return "ot";
        }
        return field;
    }

    void print_header(const cmd_params & params) override {
        // select fields to print
        fields.emplace_back("model");
        fields.emplace_back("size");
        fields.emplace_back("params");
        fields.emplace_back("backend");
        bool is_cpu_backend = test::get_backend().find("CPU") != std::string::npos ||
                              test::get_backend().find("BLAS") != std::string::npos;
        if (!is_cpu_backend) {
            fields.emplace_back("n_gpu_layers");
        }
        if (params.n_threads.size() > 1 || params.n_threads != cmd_params_defaults.n_threads || is_cpu_backend) {
            fields.emplace_back("n_threads");
        }
        if (params.cpu_mask.size() > 1 || params.cpu_mask != cmd_params_defaults.cpu_mask) {
            fields.emplace_back("cpu_mask");
        }
        if (params.cpu_strict.size() > 1 || params.cpu_strict != cmd_params_defaults.cpu_strict) {
            fields.emplace_back("cpu_strict");
        }
        if (params.poll.size() > 1 || params.poll != cmd_params_defaults.poll) {
            fields.emplace_back("poll");
        }
        if (params.n_batch.size() > 1 || params.n_batch != cmd_params_defaults.n_batch) {
            fields.emplace_back("n_batch");
        }
        if (params.n_ubatch.size() > 1 || params.n_ubatch != cmd_params_defaults.n_ubatch) {
            fields.emplace_back("n_ubatch");
        }
        if (params.type_k.size() > 1 || params.type_k != cmd_params_defaults.type_k) {
            fields.emplace_back("type_k");
        }
        if (params.type_v.size() > 1 || params.type_v != cmd_params_defaults.type_v) {
            fields.emplace_back("type_v");
        }
        if (params.defrag_thold.size() > 1 || params.defrag_thold != cmd_params_defaults.defrag_thold) {
            fields.emplace_back("defrag_thold");
        }
        if (params.main_gpu.size() > 1 || params.main_gpu != cmd_params_defaults.main_gpu) {
            fields.emplace_back("main_gpu");
        }
        if (params.split_mode.size() > 1 || params.split_mode != cmd_params_defaults.split_mode) {
            fields.emplace_back("split_mode");
        }
        if (params.no_kv_offload.size() > 1 || params.no_kv_offload != cmd_params_defaults.no_kv_offload) {
            fields.emplace_back("no_kv_offload");
        }
        if (params.flash_attn.size() > 1 || params.flash_attn != cmd_params_defaults.flash_attn) {
            fields.emplace_back("flash_attn");
        }
        if (params.tensor_split.size() > 1 || params.tensor_split != cmd_params_defaults.tensor_split) {
            fields.emplace_back("tensor_split");
        }
        if (params.tensor_buft_overrides.size() > 1 || !vec_vec_tensor_buft_override_equal(params.tensor_buft_overrides, cmd_params_defaults.tensor_buft_overrides)) {
            fields.emplace_back("tensor_buft_overrides");
        }
        if (params.use_mmap.size() > 1 || params.use_mmap != cmd_params_defaults.use_mmap) {
            fields.emplace_back("use_mmap");
        }
        if (params.embeddings.size() > 1 || params.embeddings != cmd_params_defaults.embeddings) {
            fields.emplace_back("embeddings");
        }
        if (params.no_op_offload.size() > 1 || params.no_op_offload != cmd_params_defaults.no_op_offload) {
            fields.emplace_back("no_op_offload");
        }
        fields.emplace_back("test");
        fields.emplace_back("t/s");

        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", get_field_width(field), get_field_display_name(field).c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field : fields) {
            int width = get_field_width(field);
            fprintf(fout, " %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        fprintf(fout, "\n");
    }

    void print_test(const test & t) override {
        std::map<std::string, std::string> vmap = t.get_map();

        fprintf(fout, "|");
        for (const auto & field : fields) {
            std::string value;
            char        buf[128];
            if (field == "model") {
                value = t.model_type;
            } else if (field == "size") {
                snprintf(buf, sizeof(buf), "%.2f MiB", t.model_size / 1024.0 / 1024.0);
                value = buf;
            } else if (field == "params") {
                snprintf(buf, sizeof(buf), "%.2f M", t.model_n_params / 1e6);
                value = buf;
            } else if (field == "backend") {
                value = test::get_backend();
            } else if (field == "test") {
                if (t.n_prompt > 0 && t.n_gen == 0) {
                    snprintf(buf, sizeof(buf), "pp%d", t.n_prompt);
                } else if (t.n_gen > 0 && t.n_prompt == 0) {
                    snprintf(buf, sizeof(buf), "tg%d", t.n_gen);
                } else {
                    snprintf(buf, sizeof(buf), "pp%d+tg%d", t.n_prompt, t.n_gen);
                }
                if (t.n_depth > 0) {
                    int len = strlen(buf);
                    snprintf(buf + len, sizeof(buf) - len, " @ d%d", t.n_depth);
                }
                value = buf;
            } else if (field == "t/s") {
                snprintf(buf, sizeof(buf), "%.2f", t.model_ts());
                value = buf;
            } else if (vmap.find(field) != vmap.end()) {
                value = vmap.at(field);
            } else {
                assert(false);
                exit(1);
            }

            int width = get_field_width(field);
            if (field == "t/s") {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");

        for (auto layer: t.layers) {
            fprintf(fout, "|");
            struct Stats stat = g_collector.get_layer(layer.name);
            for (const auto & field : fields) {
                std::string value;
                char        buf[128];
                if (field == "model") {
                    value = layer.name;
                } else if (field == "size") {
                    snprintf(buf, sizeof(buf), "%.2f MiB", stat.size / 1024.0 / 1024.0);
                    value = buf;
                } else if (field == "params") {
                    snprintf(buf, sizeof(buf), "%.2f M", stat.n_params / 1e6);
                    value = buf;
                } else if (field == "backend") {
                    value = test::get_backend();
                } else if (field == "test") {
                    if (t.n_prompt > 0 && t.n_gen == 0) {
                        snprintf(buf, sizeof(buf), "pp%d", t.n_prompt);
                    } else if (t.n_gen > 0 && t.n_prompt == 0) {
                        snprintf(buf, sizeof(buf), "tg%d", t.n_gen);
                    } else {
                        snprintf(buf, sizeof(buf), "pp%d+tg%d", t.n_prompt, t.n_gen);
                    }
                    if (t.n_depth > 0) {
                        int len = strlen(buf);
                        snprintf(buf + len, sizeof(buf) - len, " @ d%d", t.n_depth);
                    }
                    value = buf;
                } else if (field == "t/s") {
                    snprintf(buf, sizeof(buf), "%.2f", layer.ts);
                    value = buf;
                } else if (vmap.find(field) != vmap.end()) {
                    value = vmap.at(field);
                } else {
                    assert(false);
                    exit(1);
                }

                int width = get_field_width(field);
                if (field == "t/s") {
                    // HACK: the utf-8 character is 2 bytes
                    width += 1;
                }
                fprintf(fout, " %*s |", width, value.c_str());
            }
            fprintf(fout, "\n");
        }
    }
};

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    if (num_digits > 128) num_digits = 128;

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    setpriority(PRIO_PROCESS, 0, p);
    return true;
}

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

int32_t cpu_get_num_physical_cores() {
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t cpu_get_num_math() {
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
    return 0;
}

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS GGML_MAX_SRC
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][1];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;
};
struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

bool floatArraysEqual(const float* arr1, const float* arr2, size_t size, float epsilon = 1e-6f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

bool ActCollector::collect_activations(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (m_stats.find(wname) != m_stats.end()) {
            struct Stats* stat = &g_collector.get_layer(wname);
            stat->n_params = src0->ne[0]*src0->ne[1];
            stat->size = src0->nb[2];
            return true;
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t src1_nbytes = ggml_nbytes(src1);
    const size_t data_nbytes = ggml_nbytes(t);
    m_stats[wname].input_act.resize(src1_nbytes);
    m_stats[wname].output_act.resize(data_nbytes);
    // copy the data from the GPU memory if needed
    if (!ggml_backend_buffer_is_host(src1->buffer)) {
        ggml_backend_tensor_get(src1, m_stats[wname].input_act.data(), 0, src1_nbytes);
    }
    else {
        memcpy(m_stats[wname].input_act.data(), (const char *) src1->data, src1_nbytes);
    }

    if (!ggml_backend_buffer_is_host(t->buffer)) {
        ggml_backend_tensor_get(t, m_stats[wname].output_act.data(), 0, data_nbytes);
    }
    else {
        memcpy(m_stats[wname].output_act.data(), (const char *) t->data, data_nbytes);
    }

    return true;
}
