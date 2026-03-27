#include "net.h"
#include "cpu.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <dirent.h>

#define VAL_DATA_PATH "/pytorch/prune/imagenetv2/imagenetv2-matched-frequency-format-val/"
// ResNet50 官方预处理参数
const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
const float norm_vals[3] = {0.01712475f, 0.017507f, 0.01742919f};
// 1. 定义目标尺寸
const int target_short_side = 256;
const int target_crop_size = 224;

int main(int argc, char** argv) {
    //if (argc != 2) {
    //    printf("Usage: %s <imagenet-v2-path>\n", argv[0]);
    //    return -1;
    //}

    ncnn::Net resnet;
#if 0
    //resnet.load_param("/pytorch/prune/ncnn/resnet50_pruned_v2.ncnn.param");
    //resnet.load_model("/pytorch/prune/ncnn/resnet50_pruned_v2.ncnn.bin");
    resnet.load_param("/pytorch/prune/ncnn/resnet50_opt.param");
    resnet.load_model("/pytorch/prune/ncnn/resnet50_opt.bin");
#else
    resnet.load_param("/pytorch/ncnn/build/onnx2ncnn/model.ncnn.param");
    resnet.load_model("/pytorch/ncnn/build/onnx2ncnn/model.ncnn.bin");
#endif
    ncnn::set_omp_num_threads(8);
    int total = 0, correct = 0;
    // 遍历 0-999 文件夹
    for (int cls = 0; cls < 1000; cls++) {
        std::string cls_dir = std::string(VAL_DATA_PATH) + "/" + std::to_string(cls);
        //std::string cls_dir = std::string(argv[1]) + "/" + std::to_string(cls);
        DIR* dir = opendir(cls_dir.c_str());
        if (!dir) continue;

        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;

            std::string img_path = cls_dir + "/" + entry->d_name;
            cv::Mat bgr = cv::imread(img_path);
            if (bgr.empty()) continue;
#if 0
            // 预处理：Resize(256) -> CenterCrop(224)
            // ncnn 内部 from_pixels_resize 效率很高
            ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, 
                ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
            in.substract_mean_normalize(mean_vals, norm_vals);
            ncnn::Extractor ex = resnet.create_extractor();
            //ex.set_num_threads(4); // 设置 CPU 核心数
            ncnn::Mat out;
            ex.input("in0", in);
#else
	    int width = bgr.cols;
            int height = bgr.rows;
	    int w2, h2;
            if (width < height) {
                w2 = target_short_side;
                h2 = (int)((float)height * target_short_side / width);
            } else {
                h2 = target_short_side;
                w2 = (int)((float)width * target_short_side / height);
            }
            
            // 2. 先缩放到短边 256 (注意这里输出的是 in_resized)
            ncnn::Mat in_resized = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w2, h2);
            
            // 3. 计算裁剪边界 (从 w2 x h2 中裁出 224 x 224)
            int x = (w2 - target_crop_size) / 2;
            int y = (h2 - target_crop_size) / 2;
            
            // ncnn::copy_cut_border 的参数含义是：src, dst, top, bottom, left, right (注意是切掉的像素数)
            // 我们要保留中间 224，所以切掉四周
            ncnn::Mat in_cropped;
            ncnn::copy_cut_border(in_resized, in_cropped, y, h2 - y - target_crop_size, x, w2 - x - target_crop_size);
            // 归一化
            in_cropped.substract_mean_normalize(mean_vals, norm_vals);
            ncnn::Extractor ex = resnet.create_extractor();
            //ex.set_num_threads(4); // 设置 CPU 核心数
            ncnn::Mat out;
            ex.input("in0", in_cropped);
#endif
            ex.extract("out0", out);

            // 获取 Top-1
            int pred = -1;
            float max_p = -1e9;
            for (int j = 0; j < out.w; j++) {
                if (out[j] > max_p) { max_p = out[j]; pred = j; }
            }

            if (pred == cls) correct++;
            total++;
        }
        closedir(dir);
        if (cls % 100 == 0) printf("Tested %d classes...\n", cls);
    }

    printf("\n[Final Result] Accuracy: %.2f%% (%d/%d)\n", 
           (float)correct/total*100, correct, total);
    return 0;
}

