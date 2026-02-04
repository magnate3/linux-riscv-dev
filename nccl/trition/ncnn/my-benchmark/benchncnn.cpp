// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause
// The basic code comes from ncnn warehouse.
// https://github.com/Tencent/ncnn/blob/master/benchmark/benchncnn.cpp

#include <cfloat>
#include <cstdio>
#include <cstring>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#ifdef DEBUG_MODE
#include "ncnn/benchmark.h"
#include "ncnn/cpu.h"
#include "ncnn/datareader.h"
#include "ncnn/net.h"
#include "ncnn/layer.h"
#include "ncnn/gpu.h"
#else

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "layer.h"
#include "gpu.h"

#endif


#ifndef NCNN_SIMPLESTL
#include <vector>
#endif

enum TaskType : int {
    TEST = -1,
    ALL = 0,
    YOLOV5 = 1,
    YOLOV6 = 2,
    YOLOV7 = 3,
    YOLOV8 = 4,
    YOLOV9 = 5,
    YOLOV10 = 6,
    YOLO11 = 7,
    YOLOV12 = 8,
    YOLOV13 = 9,
    YOLOX = 10,
    YOLO26 = 11,
};

TaskType string2TaskType(const char *str) {
    if (strcmp(str, "all") == 0) {
        return TaskType::ALL;
    } else if (strcmp(str, "test") == 0) {
        return TaskType::TEST;
    } else if (strcmp(str, "yolov5") == 0) {
        return TaskType::YOLOV5;
    } else if (strcmp(str, "yolov6") == 0) {
        return TaskType::YOLOV6;
    } else if (strcmp(str, "yolov7") == 0) {
        return TaskType::YOLOV7;
    } else if (strcmp(str, "yolov8") == 0) {
        return TaskType::YOLOV8;
    } else if (strcmp(str, "yolov9") == 0) {
        return TaskType::YOLOV9;
    } else if (strcmp(str, "yolov10") == 0) {
        return TaskType::YOLOV10;
    } else if (strcmp(str, "yolov11") == 0 || strcmp(str, "yolo11") == 0) {
        return TaskType::YOLO11;
    } else if (strcmp(str, "yolov12") == 0) {
        return TaskType::YOLOV12;
    } else if (strcmp(str, "yolov13") == 0) {
        return TaskType::YOLOV13;
    } else if (strcmp(str, "yolox") == 0) {
        return TaskType::YOLOX;
    } else if (strcmp(str, "yolov26") == 0 || strcmp(str, "yolo26") == 0) {
        return TaskType::YOLO26;
    }
    return TaskType::ALL;
}

// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer {
public:
    YoloV5Focus() {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++) {
            const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

class DataReaderFromEmpty : public ncnn::DataReader {
public:
    virtual int scan(const char *format, void *p) const {
        return 0;
    }

    virtual size_t read(void *buf, size_t size) const {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void
benchmark(const std::string &base_path, const char *comment, const std::vector<ncnn::Mat> &_in, const ncnn::Option &opt,
          bool fixed_path = true) {
    // Skip if int8 model name and using GPU
    if (opt.use_vulkan_compute && strstr(comment, "int8") != NULL) {
        if (!fixed_path)
            fprintf(stderr, "%20s  skipped (int8+GPU not supported)\n", comment);
        return;
    }

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;
    net.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    if (fixed_path) {
        char parampath[1024];
        sprintf(parampath, (base_path + "%s.param").c_str(), comment);
        net.load_param(parampath);
    } else {
        net.load_param(comment);
    }

    DataReaderFromEmpty dr;
    net.load_model(dr);

    const std::vector<const char *> &input_names = net.input_names();
    const std::vector<const char *> &output_names = net.output_names();

    if (g_enable_cooling_down) {
        // sleep 10 seconds for cooling down SOC  :(
        ncnn::sleep(10 * 1000);
    }

    if (input_names.size() > _in.size()) {
        fprintf(stderr, "input %zu tensors while model has %zu inputs\n", _in.size(), input_names.size());
        return;
    }

    // initialize input
    for (size_t j = 0; j < input_names.size(); ++j) {
        ncnn::Mat in = _in[j];
        in.fill(0.01f);
    }

    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++) {
        ncnn::Extractor ex = net.create_extractor();
        for (size_t j = 0; j < input_names.size(); ++j) {
            ncnn::Mat in = _in[j];
            ex.input(input_names[j], in);
        }

        for (size_t j = 0; j < output_names.size(); ++j) {
            ncnn::Mat out;
            ex.extract(output_names[j], out);
        }
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++) {
        double start = ncnn::get_current_time();
        {
            ncnn::Extractor ex = net.create_extractor();
            for (size_t j = 0; j < input_names.size(); ++j) {
                ncnn::Mat in = _in[j];
                ex.input(input_names[j], in);
            }

            for (size_t j = 0; j < output_names.size(); ++j) {
                ncnn::Mat out;
                ex.extract(output_names[j], out);
            }
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
}

void benchmark(const std::string &base_path, const char *comment, const ncnn::Mat &_in, const ncnn::Option &opt,
               bool fixed_path = true) {
    std::vector<ncnn::Mat> inputs;
    inputs.push_back(_in);
    return benchmark(base_path, comment, inputs, opt, fixed_path);
}

void show_usage() {
    fprintf(stderr,
            "Usage: benchncnn [task type] [loop count] [num threads] [powersave] [gpu device] [cooling down] [(key=value)...]\n");
    fprintf(stderr, "  param=model.param\n");
    fprintf(stderr, "  shape=[227,227,3],...\n");
}

static std::vector<ncnn::Mat> parse_shape_list(char *s) {
    std::vector<std::vector<int> > shapes;
    std::vector<ncnn::Mat> mats;

    char *pch = strtok(s, "[]");
    while (pch != NULL) {
        // parse a,b,c
        int v;
        int nconsumed = 0;
        int nscan = sscanf(pch, "%d%n", &v, &nconsumed);
        if (nscan == 1) {
            // ok we get shape
            pch += nconsumed;

            std::vector<int> s;
            s.push_back(v);

            nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            while (nscan == 1) {
                pch += nconsumed;

                s.push_back(v);

                nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            }

            // shape end
            shapes.push_back(s);
        }

        pch = strtok(NULL, "[]");
    }

    for (size_t i = 0; i < shapes.size(); ++i) {
        const std::vector<int> &shape = shapes[i];
        switch (shape.size()) {
            case 4:
                mats.push_back(ncnn::Mat(shape[0], shape[1], shape[2], shape[3]));
                break;
            case 3:
                mats.push_back(ncnn::Mat(shape[0], shape[1], shape[2]));
                break;
            case 2:
                mats.push_back(ncnn::Mat(shape[0], shape[1]));
                break;
            case 1:
                mats.push_back(ncnn::Mat(shape[0]));
                break;
            default:
                fprintf(stderr, "unsupported input shape size %zu\n", shape.size());
                break;
        }
    }
    return mats;
}

int main(int argc, char **argv) {
    TaskType taskType = TaskType::ALL;
    int loop_count = 4;
    int num_threads = ncnn::get_physical_big_cpu_count();
    int powersave = 2;
    int gpu_device = -1;
    int cooling_down = 1;
    char *model = 0;
    std::vector<ncnn::Mat> inputs;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == 'h') {
            show_usage();
            return -1;
        }

        if (strcmp(argv[i], "--help") == 0) {
            show_usage();
            return -1;
        }
    }

    if (argc >= 2) {
        taskType = string2TaskType(argv[1]);
    }
    if (argc >= 3) {
        loop_count = atoi(argv[2]);
    }
    if (argc >= 4) {
        num_threads = atoi(argv[3]);
    }
    if (argc >= 5) {
        powersave = atoi(argv[4]);
    }
    if (argc >= 6) {
        gpu_device = atoi(argv[5]);
    }
    if (argc >= 7) {
        cooling_down = atoi(argv[6]);
    }

    for (int i = 7; i < argc; i++) {
        // key=value
        char *kv = argv[i];

        char *eqs = strchr(kv, '=');
        if (eqs == NULL) {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            continue;
        }

        // split k v
        eqs[0] = '\0';
        const char *key = kv;
        char *value = eqs + 1;

        if (strcmp(key, "param") == 0)
            model = value;
        if (strcmp(key, "shape") == 0)
            inputs = parse_shape_list(value);
    }

    if (model && inputs.empty()) {
        fprintf(stderr, "input tensor shape empty!\n");
        return -1;
    }

#ifdef __EMSCRIPTEN__
    EM_ASM(
        FS.mkdir('/working');
        FS.mount(NODEFS, {root: '.'}, '/working'););
#endif // __EMSCRIPTEN__

    bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    // default option
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;

    fprintf(stderr, "task type = %d\n", (int) taskType);
    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int) g_enable_cooling_down);

    if (model != 0 || taskType == TaskType::TEST) {
        // run user defined benchmark
        benchmark("", model, inputs, opt, false);
    } else {
        // run default cases
        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV5) {
            std::string base_path = "../params/yolov5/";
            benchmark(base_path, "yolov5n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov5s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov5m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov5l", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov5x", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV6) {
            std::string base_path = "../params/yolov6/";
            benchmark(base_path, "yolov6n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov6s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov6m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov6l", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV7) {
            std::string base_path = "../params/yolov7/";
            benchmark(base_path, "yolov7t", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov7", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov7x", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV8) {
            std::string base_path = "../params/yolov8/";
            benchmark(base_path, "yolov8n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov8s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov8m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov8l", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov8x", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV9) {
            std::string base_path = "../params/yolov9/";
            benchmark(base_path, "yolov9t", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov9s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov9m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov9c", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov9e", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV10) {
            std::string base_path = "../params/yolov10/";
            benchmark(base_path, "yolov10n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov10s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov10m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov10b", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov10l", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov10x", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLO11) {
            std::string base_path = "../params/yolo11/";
            benchmark(base_path, "yolo11n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo11s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo11m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo11l", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo11x", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV12) {
            std::string base_path = "../params/yolov12/";
            benchmark(base_path, "yolov12n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov12s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov12m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov12l", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov12x", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOV13) {
            std::string base_path = "../params/yolov13/";
            benchmark(base_path, "yolov13n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov13s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov13l", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolov13x", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLOX) {
            std::string base_path = "../params/yolox/";
            benchmark(base_path, "yoloxs", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yoloxm", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yoloxl", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yoloxx", ncnn::Mat(640, 640, 3), opt);
        }

        if (taskType == TaskType::ALL || taskType == TaskType::YOLO26) {
            std::string base_path = "../params/yolo26/";
            benchmark(base_path, "yolo26n", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo26s", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo26m", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo26l", ncnn::Mat(640, 640, 3), opt);
            benchmark(base_path, "yolo26x", ncnn::Mat(640, 640, 3), opt);
        }
    }
#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
