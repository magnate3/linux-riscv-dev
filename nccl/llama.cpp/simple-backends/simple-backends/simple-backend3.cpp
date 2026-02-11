
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <map>
//#include "ggml.h"
//#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
std::vector<ggml_backend_dev_t> devices;
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a {};
    struct ggml_tensor * b {};
    struct ggml_tensor * c {};
    struct ggml_tensor * d {};
    struct ggml_context * cpu_ctx;
    struct ggml_context * gpu_ctx;
    struct ggml_context * compute_ctx;

    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t gpu_backend {};
    ggml_backend_t cpu_backend {};
    ggml_backend_sched_t sched {};
    ggml_backend_buffer_type_t cpu_buft;
    ggml_backend_buffer_type_t gpu_buft;

    // storage for the graph and tensors
    std::vector<uint8_t> buf;
    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;
};

// initialize data of matrices to perform matrix multiplication
const int rows_A = 4, cols_A = 2;

float matrix_A[rows_A * cols_A] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};

const int rows_B = 3, cols_B = 2;
/* Transpose([
    10, 9, 5,
    5, 9, 4
]) 2 rows, 3 cols */
float matrix_B[rows_B * cols_B] = {
    10, 5,
    9, 9,
    5, 4
};

const int rows_C = 3, cols_C = 4;
float matrix_C[cols_C * rows_C] = {
    10, 5,
    10, 5,
    9, 9,
    9, 8,
    9, 7,
    5, 4
};
// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev) {
    //TBD: chech if the dev is gpu??
    buft_list_t buft_list;
    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    return buft_list;
}


// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    // for (auto * dev : devices) {
    //     ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    //     if (buft) {
    //         buft_list.emplace_back(dev, buft);
    //         break;
    //     }
    // }

    // // add extra buffer types, only if no GPU device is present
    // // ref: https://github.com/ggml-org/llama.cpp/issues/12481#issuecomment-2743136094
    // auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    // auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    // auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
    //     ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    // if (ggml_backend_dev_get_extra_bufts_fn) {
    //     ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
    //     while (extra_bufts && *extra_bufts) {
    //         buft_list.emplace_back(cpu_dev, *extra_bufts);
    //         ++extra_bufts;
    //     }
    // }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }
    return buft_list;
}
// initialize the tensors of the model in this case two matrices 2x2
void init_model(simple_model & model) {
      // use scheduler
    //std::vector<ggml_backend_t> backends = {backend};	
    //std::vector<ggml_backend_t> backends;	
    std::vector<ggml_backend_ptr> backends;
    ggml_backend_t backend_cpu = nullptr;
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size*2,
        /*.mem_buffer =*/ NULL,
        ///*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later
    };

    struct ggml_init_params params1 = {
        /*.mem_size   =*/ 512+128+64+32,
        /*.mem_buffer =*/ NULL,
        ///*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later
    };
    // create a context to build the graph
    struct ggml_context * cpu_ctx = ggml_init(params0);
    model.cpu_ctx = cpu_ctx;
    struct ggml_context * gpu_ctx = ggml_init(params1);
    model.gpu_ctx = gpu_ctx;
    ggml_log_set(ggml_log_callback_default, nullptr);

    ggml_backend_load_all();

    //model.backend = ggml_backend_init_best();
    model.gpu_backend = ggml_backend_cuda_init(0);
    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    //ggml_backend_t backends2[2] = { model.cpu_backend, model.gpu_backend };
    ggml_backend_t backends2[2] = { model.gpu_backend, model.cpu_backend };
    std::vector<ggml_backend_buffer_type_t> backend_bufts2;
#if 0
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t gpu_type = 
    ggml_backend_buffer_type_t buft2 =  ggml_backend_alloc_ctx_tensors_from_buft(ctx,  ggml_backend_dev_buffer_type(GGML_BACKEND_DEVICE_TYPE_GPU));
    //ggml_backend_buffer_type_t buft2 = ggml_backend_gpu_buffer_type();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft2);
    ggml_backend_buffer_clear(buf, 0);
    ggml_backend_buffer_clear(buf2, 0);
#else
     printf("Handling Override Tensors for %ld  backends \n",ggml_backend_dev_count());
     std::vector<ggml_backend_buffer_type_t> backend_bufts;
     //std::map<std::string, ggml_backend_buffer_type_t> buft_list;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                //auto *      dev  = ggml_backend_dev_get(i);
	        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                auto *      buft = ggml_backend_dev_buffer_type(dev);
                if (buft) {
		    if(buft == ggml_backend_cpu_buffer_type())
	            {
			   model.cpu_buft = buft;
		    }
		    else
		    {
                           printf("++++++++++++ backend gpu buft \n");
			   model.gpu_buft = buft;
		    }
                    std::string name = ggml_backend_buft_name(buft);
                    printf("backend %s \n", name.c_str());
                    //buft_list[name] = buft;
		    backend_bufts.push_back(buft);
                }
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU:
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    devices.push_back(dev);
                    break;
            }
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu_dev = nullptr;
    model.cpu_buft_list = make_cpu_buft_list(devices);
        //TBD: need to chech if the dev is  gpu??? //add gpu device
    for (auto * dev : devices) {
        gpu_dev = devices.at(0);//use the fist device as normally
        buft_list_t buft_list = make_gpu_buft_list(dev);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), model.cpu_buft_list.begin(), model.cpu_buft_list.end());
        model.gpu_buft_list.emplace(dev, std::move(buft_list));
    }
#if 0
        //init teh backend
        // GPU backends
    for (auto * dev : devices) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            throw std::runtime_error("failed to initialize %s backend");
        }
        backends.emplace_back(backend);
    }

        // add CPU backend
    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_cpu == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    backends.emplace_back(backend_cpu);
        // create a list of the set_n_threads functions in the backends
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                model.set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
            }
        }
    }
#endif
#endif
    //model.sched = ggml_backend_sched_new(backends, backend_bufts.data(), 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
#if 1
    backend_bufts2.emplace_back(model.gpu_buft);
    backend_bufts2.emplace_back(model.cpu_buft);
#else
    backend_bufts2.emplace_back(model.cpu_buft);
    backend_bufts2.emplace_back(model.gpu_buft);
#endif
    model.sched = ggml_backend_sched_new(backends2,backend_bufts2.data(), 2, 32, true, true);
    //model.sched = ggml_backend_sched_new(backends2,backend_bufts.data(), 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
}
void print_buft(ggml_backend_buffer_t buf){
     size_t size_main = ggml_backend_buffer_get_size(buf);
     printf("%s: %12s total size = %lu MB\n", __func__, ggml_backend_buffer_name(buf), size_main);
     //printf("%s: %12s total size = %8.2f MB\n", __func__, ggml_backend_buffer_name(buf), size_main / 1e6);
}
int graph_max_nodes() {
    //return std::max<int32_t>(32,2);
    return 8;
}
static std::string format_tensor_shape(const std::vector<int64_t> &ne)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++)
    {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
    }
    return buf;
}


static std::string format_tensor_shape(const struct ggml_tensor *t)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++)
    {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
    }
    return buf;
}
// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(simple_model& model) {

    struct ggml_context * cpu_ctx = model.cpu_ctx;
    struct ggml_context * gpu_ctx = model.gpu_ctx;
    //std::vector<ggml_backend_buffer_type_t> backend_bufts2;
    //ggml_backend_t backends2[2] = { model.gpu_backend, model.cpu_backend };
    //std::vector<ggml_backend_buffer_type_t> backend_bufts2;
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ buf_size*2,
        /*.mem_buffer =*/ NULL,
        ///*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * compute_ctx = ggml_init(params);
    struct ggml_cgraph  * gf =  ggml_new_graph_custom(compute_ctx, graph_max_nodes(), false);
    //struct ggml_cgraph  * gf = ggml_new_graph(compute_ctx);
#if 1
     // create tensors
     //gpu_ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.gpu_backend));
     model.a = ggml_new_tensor_2d(gpu_ctx, GGML_TYPE_F32, cols_A, rows_A);
     model.b = ggml_new_tensor_2d(gpu_ctx, GGML_TYPE_F32, cols_B, rows_B);
     ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors_from_buft(gpu_ctx, model.gpu_buft);
     if(buf1 == nullptr)
     {
	     throw std::runtime_error(("unable to allocate buffer1"));
     }
     print_buft(buf1);
     ggml_backend_buffer_set_usage(buf1, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
     //model.c = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32,  rows_A,cols_B);
     model.c = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, cols_C, rows_C);
     model.d = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, cols_C, rows_C);
     //model.d = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, rows_C, cols_C);
     //model.c = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, 1);
     ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors_from_buft(cpu_ctx, model.cpu_buft);
     if(buf2 == nullptr)
     {
	     throw std::runtime_error(("unable to allocate buffer2"));
     }
     //backend_bufts2.emplace_back(buf1);
     //backend_bufts2.emplace_back(buf2);
     //model.sched = ggml_backend_sched_new(backends2,backend_bufts2.data(), 2, 32, false, true);
#if 0
     ggml_backend_buffer_set_usage(buf2, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
#endif
     print_buft(buf2);
     //ggml_backend_buffer_clear(buf1, 0);
     //ggml_backend_buffer_clear(buf2, 0);
     //ggml_backend_tensor_set(model.a, buf1->data(), 0, ggml_nbytes(model.a));
    //model.sched = ggml_backend_sched_new({model.cpu_backend,model.gpu_backend}, {model.cpu_buft,model.gpu_buft}, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
#endif
    // ggml_backend_alloc_ctx_tensors_from_buft
    // result = a*b^T
    //struct ggml_tensor * result = ggml_mul(compute_ctx, model.a, model.b);
    struct ggml_tensor * result = ggml_mul_mat(compute_ctx, model.a, model.b);
    ggml_set_name(result, "mul_mat_0");
    ggml_build_forward_expand(gf, result);
    printf("model.a shape %s \n", format_tensor_shape(model.a).data());
    printf("model.b shape %s \n", format_tensor_shape(model.b).data());
    printf("result shape %s \n", format_tensor_shape(result).data());
    printf("model.c shape %s \n", format_tensor_shape(model.c).data());
    printf("model.d shape %s \n", format_tensor_shape(model.d).data());
    ggml_backend_sched_set_tensor_backend(model.sched,model.a,model.gpu_backend);
    ggml_backend_sched_set_tensor_backend(model.sched,model.b,model.gpu_backend);
    ggml_backend_sched_set_tensor_backend(model.sched,model.c,model.cpu_backend);
    ggml_backend_sched_set_tensor_backend(model.sched,model.d,model.cpu_backend);
    ggml_backend_t next_bk = ggml_backend_sched_get_tensor_backend(model.sched, model.a);
    printf("model.a Backend type: %s\n", ggml_backend_name(next_bk));
    next_bk = ggml_backend_sched_get_tensor_backend(model.sched, model.b);
    printf("model.b Backend type: %s\n", ggml_backend_name(next_bk));
    next_bk = ggml_backend_sched_get_tensor_backend(model.sched, model.c);
    printf("model.c Backend type: %s\n", ggml_backend_name(next_bk));
    //struct ggml_tensor *out = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, result));
    //result = ggml_add(compute_ctx, out, model.c);
    struct ggml_tensor * result2 = ggml_add(compute_ctx, model.c, model.d);
    ggml_set_name(result2, "add_0");
    ggml_build_forward_expand(gf, result2);
    //struct ggml_tensor * result2 = ggml_mul_mat(compute_ctx, model.c, model.d);
    //ggml_set_name(result2, "mul_mat_1");
    result = ggml_add(compute_ctx, result, result2);
    //result = ggml_add(compute_ctx, result, model.c);
    ggml_set_name(result, "add_1");
    //result = ggml_mul_mat(compute_ctx, result, model.c);
    //result = ggml_mul(compute_ctx, model.c, result);
    //ggml_set_name(result, "mul_mat_1");

    // build operations nodes
    ggml_build_forward_expand(gf, result);


    model.compute_ctx = compute_ctx;
    ggml_graph_node(gf, -1);
    return gf;
}
static bool ggml_graph_compute_helper(
      ggml_backend_sched_t   sched,
        struct ggml_cgraph * graph,
                       int   n_threads) {

        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

            auto * fn_set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn_set_n_threads) {
                fn_set_n_threads(backend, n_threads);
            }
        }

        bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_sched_reset(sched);
        return t;
}
static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads,NULL);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}
//void llama_attach_threadpool(struct ggml_context * ctx, ggml_threadpool_t threadpool,
//                             ggml_threadpool_t threadpool_batch) {
//    ctx->threadpool       = threadpool;
//    ctx->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
//}
// compute with backend
struct ggml_tensor * compute(simple_model & model, struct ggml_cgraph * gf) {
#if 1
    //ggml_backend_sched_reset(model.sched);
    //printf("sched reset \n");
    //printf("%d nodes \n",  ggml_graph_n_nodes(gf));
    ggml_backend_sched_alloc_graph(model.sched, gf);

    printf("tensor set \n");
    //ggml_graph_compute_helper(model.sched, gf,2);
#else
    //ggml_backend_sched_reset(model.sched);
    //ggml_backend_sched_alloc_graph(model.sched, gf);
    ggml_backend_sched_reserve(model.sched, gf);
    printf("sched reserve\n");
#endif
#if 1
    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));
    ggml_backend_tensor_set(model.c, matrix_C, 0, ggml_nbytes(model.c));
    ggml_backend_tensor_set(model.d, matrix_C, 0, ggml_nbytes(model.d));
    //ggml_set_f32(model.c, 3.0f);
#else
    //if(tensor->backend != GGML_BACKEND_GPU)
#endif
#if 1
     for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            ggml_backend_dev_t device_fa = ggml_backend_get_device(
                    ggml_backend_sched_get_tensor_backend(model.sched, n));
	    printf("\n !!!!!!! node %s on  device %s \n",ggml_get_name(n),ggml_backend_dev_name(device_fa));
    }
     for (int i = 0; i < ggml_backend_sched_get_n_backends(model.sched); ++i) 
     {
        ggml_backend_t backend = ggml_backend_sched_get_backend(model.sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
	printf("\n !!!!!!! backend  on  device %s \n",ggml_backend_dev_name(dev));

    }
    //for (int i = 0; i < ggml_graph_n_leafs(gf); i++) {
    //      // ggml_tensor_t* leaf = gf->leafs[i];
    //      ggml_tensor * leaf = ggml_graph_leaf(gf, i);
    //        ggml_backend_dev_t device_fa = ggml_backend_get_device(
    //                ggml_backend_sched_get_tensor_backend(model.sched, leaf));
    //        printf("\n !!!!!!! leaf device %s \n",ggml_backend_dev_name(device_fa));
    //}
    //ggml_backend_sched_split_graph(model.sched, gf);
    //struct ggml_backend_sched_split * splits = model.sched->splits;

#endif
#if 0
    for (int i = 0; i < model.sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &splits[i];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
     printf("%d",model.sched->n_splits);
   }
#endif
#if 0
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        fprintf(stderr, "%s: error: CPU backend is not loaded\n", __func__);
        exit(1);
    }
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(8);
    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        fprintf(stderr, "%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        exit(1);
    }
#endif
#if 0
     {
        struct ggml_tensor* next_out = ggml_graph_get_tensor(gf, "add_1");
        float next_buffer[10];
        ggml_backend_t next_bk = ggml_backend_sched_get_tensor_backend(model.sched, next_out);
        printf("Backend type: %s\n", ggml_backend_name(next_bk));
        printf("Tensor type: %s\n", ggml_type_name(next_out->type));
        //ggml_backend_tensor_get_async(next_bk, next_out, next_buffer, 0, sizeof(next_buffer));
        //ggml_backend_sched_synchronize(ctx.sched);
        //for (int i = 0; i < 10; i++) {
        //    printf("layernorm_mul_copy-0[%d] = %f (isnan=%d)\n", i, next_buffer[i], std::isnan(next_buffer[i]));
        //}
    }
#endif
#if 0
     if (model.backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(model.backend_cpu, 8);
        //ggml_backend_cpu_set_abort_callback(mdoel.backend_cpu, lctx.abort_callback, lctx.abort_callback_data);
    }
#endif
    fprintf(stderr, "+++++++++++ splits: %d\n", ggml_backend_sched_get_n_splits(model.sched));
    //ggml_backend_sched_print_assignments(model.sched, gf);
    // compute the graph
    ggml_backend_sched_graph_compute(model.sched, gf);
    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}
int compute(simple_model & model){

    
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_reset(model.sched);
    struct ggml_cgraph * gf = build_graph(model);
    // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

        // add CPU backend
    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    //if (backend_cpu == nullptr) {
    //    throw std::runtime_error("failed to initialize CPU backend");
    //}
    //backends.emplace_back(backend_cpu);
    ggml_backend_sched_alloc_graph(model.sched, gf);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));
    ggml_backend_tensor_set(model.c, matrix_C, 0, ggml_nbytes(model.c));
    ggml_backend_tensor_set(model.d, matrix_C, 0, ggml_nbytes(model.d));
    int n_threads        = 10;
    ggml_threadpool_t tp = nullptr;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        set_threadpool_fn(backend_cpu, tp);
    }
#if 0
    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : model.set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }
#endif
    auto status = ggml_backend_sched_graph_compute_async(model.sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    return status;
}

int main(void) {
    ggml_time_init();

    simple_model model;
    init_model(model);
#if 1
    struct ggml_cgraph * gf = build_graph(model);
    ggml_graph_print(gf);
    printf("start to compute \n");
    // perform computation
    struct ggml_tensor * result = compute(model, gf);
    printf("compute  over\n");
    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // expected result:
    // [ 60.00 55.00 50.00 110.00
    //  90.00 54.00 54.00 126.00
    //  42.00 29.00 28.00 64.00 ]

    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");
#else
    compute(model);
#endif
    // release backend memory and free backend
    //ggml_backend_free(model.gpu_backend);
    //ggml_backend_free(model.cpu_backend);
    ggml_free(model.gpu_ctx);
    ggml_free(model.cpu_ctx);
    ggml_free(model.compute_ctx);
    ggml_backend_sched_free(model.sched);
    ggml_backend_free(model.gpu_backend);
    ggml_backend_free(model.cpu_backend);
    return 0;
}
