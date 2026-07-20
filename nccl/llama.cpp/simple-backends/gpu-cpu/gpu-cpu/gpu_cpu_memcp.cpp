#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <vector>
#include <cstdio>
const char* RED = "\033[0;31m";
const char* GREEN = "\033[0;32m";
const char* BLUE = "\033[0;34m";
const char* ORANGE = "\033[0;33m";  // Actually yellow, but often appears as orange in many terminals
const char* RESET = "\033[0m";
bool is_near(float a, float b) {
    return std::abs(a - b) < 1e-5;
}
int main() {

    auto * cpu_dev                             = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    //auto * cpu_reg                             = ggml_backend_dev_backend_reg(cpu_dev);
    ggml_backend_t backend_cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize cpu backend");
    }
    auto * gpu_dev                             = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t backend_gpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!backend_gpu) {
            throw std::runtime_error("failed to initialize gpu backend");
    }

    // 2. KV Cache 
    struct ggml_init_params params = { 1024 * 1024, NULL, true};
    //struct ggml_init_params params = { 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

     const int n_elements = 1024 * 512;
    struct ggml_tensor * kv_cpu = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    struct ggml_tensor * kv_gpu = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);

    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx, backend_cpu);
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx, backend_gpu);
    std::vector<float> src_data(n_elements);
    for (int i = 0; i < n_elements; i++) src_data[i] = (float)i * 0.1f;
    ggml_backend_tensor_set(kv_cpu, src_data.data(), 0, ggml_nbytes(kv_cpu));

    bool supports_async = false; // backend_gpu->iface.event_record != NULL
    if (supports_async) {
         ggml_backend_tensor_copy_async(backend_cpu,backend_gpu,kv_gpu, kv_cpu);
         ggml_backend_event_t copy_done_event = ggml_backend_event_new(gpu_dev);


         ggml_backend_event_record(copy_done_event,backend_gpu);
         //ggml_backend_event_wait(backend_gpu, copy_done_event);

         // 5. 执行推理 (模拟计算图)
         //struct ggml_cgraph * gf = ggml_new_graph(ctx);
         // 这里可以构建使用 kv_gpu 的计算图...
         // ggml_backend_graph_compute(backend_gpu, gf);
          ggml_backend_event_synchronize(copy_done_event);
 
         ggml_backend_event_free(copy_done_event);
    }
    else {
         //ggml_backend_tensor_copy_async(backend_cpu,backend_gpu,kv_gpu, kv_cpu);
	  ggml_backend_tensor_copy(kv_cpu, kv_gpu);
    }
    std::vector<float> dst_data(n_elements);
    ggml_backend_tensor_get(kv_gpu, dst_data.data(), 0, ggml_nbytes(kv_gpu));
    bool success = true;
    for (int i = 0; i < 10; i++) { // 抽样检查前10个元素
        if (!is_near(src_data[i], dst_data[i])) {
            success = false;
            break;
        }
    }
    if(success){
	 printf("%s cpu --> gpu success \n  %s", BLUE,  RESET);
    }
    else {
	printf("%s cpu --> gpu fail\n  %s", BLUE,  RESET);
    }
    ggml_backend_buffer_free(buf_cpu);
    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx);
    ggml_backend_free(backend_gpu);
    ggml_backend_free(backend_cpu);

    return 0;
}
