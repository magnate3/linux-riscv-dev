// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include "cpu.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>// putText()
#include <stdio.h>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <numeric>
#include <filesystem> // C++17 标准库

namespace fs = std::filesystem;

#define CV_FONT_HERSHEY_SIMPLEX         0
#include <iostream>
#include <string>
#include <fstream>
#include <regex>
#include <json/json.h>
#define UNKNOWN_CLASS -1
#define STR_UNKNOWN_CLASS "unkonw_class"
#define NUM_CLASS 200

//#define VAL_DATA_PATH "/pytorch/prune/tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/val/"
#define VAL_DATA_PATH "/pytorch/prune/imagenetv2/imagenetv2-matched-frequency-format-val/"
//#define VAL_DATA_PATH "/pytorch/prune/tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/test/"
//#define CLASS_FILE "/pytorch/prune/tinyimagenet/label_to_tiny_class.json"
//#define CLASS_FILE "./class_to_index.json"
#define CLASS_FILE "./index_to_class.json"

int config_to_json() {
  std::ifstream infile;
  infile.open("synset.txt");
  std::vector<std::string> labels;
  std::string line;
  while (getline(infile, line))
  {
    labels.push_back(line);
  }
  return 0;
}
int get_label_index(std::string &label_code)
{
	Json::Reader reader;
	Json::Value root;
        int index = UNKNOWN_CLASS;
	std::ifstream in("../class_to_index.json", std::ios::binary);

	if (!in.is_open())
	{
		std::cout << "Error opening file\n";
	}

	if (reader.parse(in, root))
	{

		if(!root[label_code].isNull()){

		     index = root[label_code].asInt();
		     //std::cout << "key<label code> " <<  label_code << "  vaule<class index> "<< root[label_code].asInt() << std::endl;
		}
	}
	else
	{
		std::cout << "parse error\n" << std::endl;
	}

	in.close();
	return index;
}
int get_label_json(Json::Value & root)
{
	Json::Reader reader;
	std::ifstream in(CLASS_FILE, std::ios::binary);
	if (!in.is_open())
	{
		std::cout << "Error opening file\n";
	        in.close();
		return -1;
	}
	if (reader.parse(in, root))
	{
	}
	else
	{
		std::cout << "parse error\n" << std::endl;
	        in.close();
		return -1;
	}
	in.close();
	return 0;
}
std::string get_label(const std::string &text) {
    //std::string text = "n02395406_hog.JPEG";
    std::regex pattern(R"(^([^_]+)_([^.]+)\.JPEG$)"); // 定义分组
    std::smatch matches;

    // 类似 re.match(pattern, text)
    if (std::regex_match(text, matches, pattern)) {
        //std::cout << "Full match: " << matches[0] << std::endl;

        //std::cout << text << "  label code: " << matches[1] << std::endl;
	return matches[1].str(); 
    } else {
        std::cout << "Match failed" << std::endl;
    }

    return std::string("");
}
int get_class_index(const std::string &text,Json::Value & root)
{
    std::string label = get_label(text); 
    int index = UNKNOWN_CLASS;
    if(label.empty())
	return UNKNOWN_CLASS;
    if(!root[label].isNull()){
         index = root[label].asInt();
         //std::cout << "key<label code> " <<  label_code << "  vaule<class index> "<< root[label_code].asInt() << std::endl;
    }
    return index;
}
std::string get_tiny_class(std::string cls_inx,Json::Value & root)
{
    std::string cls = STR_UNKNOWN_CLASS;
    if(!root[cls_inx].isNull()){
         cls = root[cls_inx].asString();
         //std::cout << "key<label code> " <<  label_code << "  vaule<class index> "<< root[label_code].asInt() << std::endl;
    }
    return cls;
}
long getTimeUsec()
{

    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}
static int detect_resnet(ncnn::Net & resnet , const cv::Mat& bgr, std::vector<float>& cls_scores)
{

                                         
    // ResNet50 官方预处理：Resize(256) -> CenterCrop(224)
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
                                         
#if 0
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);
#else
    // ImageNet 标准归一化参数
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {0.01712475f, 0.017507f, 0.01742919f};
    in.substract_mean_normalize(mean_vals, norm_vals);
#endif
                                         
//const float std_vals[3] = {57.0f, 57.0f, 58.0f};

    //const float std_vals[3] = {0.225f, 0.224f,0.229f};
    //in.substract_mean_normalize(mean_vals, std_vals);
    ncnn::Extractor ex = resnet.create_extractor();

    ex.input("in0", in);

    ncnn::Mat out;
//ex.extract("resnetv17_dense0_fwd", out);

    ex.extract("out0", out);
#if 1
    //std::cout <<"output size: "<< out.total()  << ", out.w:  " << out.w << std::endl;
    //ex.extract("resnetv24_stage4_activation8", out);
    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }
#else
    cls_scores.resize(NUM_CLASS);
    for (int j = 0; j < NUM_CLASS; j++)
    {
        cls_scores[j] = out[j];
    }
#endif
    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
    int K) {
    auto indices = std::vector<int>(score.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
        [&score](int a, int b) { return score[a] > score[b]; });
    auto ret = std::vector<std::pair<int, float>>(K);
    std::transform(
        indices.begin(), indices.begin() + K, ret.begin(),
        [&score](int index) { return std::make_pair(index, score[index]); });
    return ret;
}

std::string get_filename(const std::string& full_path) {
    // Find the last occurrence of either '/' or '\'
    size_t last_slash_pos = full_path.find_last_of("/\\");

    // If a slash was found, return the substring after it
    if (last_slash_pos != std::string::npos) {
        return full_path.substr(last_slash_pos + 1);
    }

    // Otherwise, the input string is just a filename
    return full_path;
}
int v2_traverse_subdir(ncnn::Net & resnet,Json::Value & labels, std::string sub, int& right ,int & wrong )
{
    // Vector to store the list of filenames
    std::vector<std::string> filenames;
    std::string pattern = VAL_DATA_PATH + sub + "/*.jpeg"; 
    //std::string pattern =  sub + "/images/*.JPEG"; 
    if (sub.empty())
    {
	   return -1;
    }
    // Use cv::glob to find all files matching the pattern
    cv::glob(pattern, filenames, false); // 'false' for non-recursive search
    //std::cout << "pattern: " << pattern << std::endl;
    // Vector to store the loaded images
    //std::vector<cv::Mat> images;
    size_t count = filenames.size(); // Number of files found

    if (count == 0) {
	    std::cout << "No images found in the directory!" << std::endl;
        return -1;
    }
    std::cout << "total jpeg: "  << count << std::endl;
    // Loop through the filenames and load each image
    for (unsigned int i = 0; i < count; i++) {
	cv::Mat img = cv::imread(filenames[i]); // Read the image
        //std::string filename = get_filename(filenames[i]);
        // Error handling: check if the image loaded successfully
        if (img.empty()) { 
		std::cout << "Error: Could not read image " << filenames[i] << std::endl;
            continue; // Skip to the next iteration
        }
        
        std::vector<float> cls_scores;
        long time = getTimeUsec();
        detect_resnet(resnet,img, cls_scores);
        time = getTimeUsec() - time;
        //printf("detection time: %ld ms\n",time/1000);
        //print_topk(cls_scores, 3);
	auto top1 = topk(cls_scores, 1)[0];
	//std::string predict = get_tiny_class(std::to_string(top1.first),labels);
	int predict = top1.first;
	int real =  std::stoi(sub);
	//std::string predict = std::to_string(top1.first);
#if 0
        std::cout << "  Predicted class index is " << top1.first << "  class  is " << predict << " real class " << sub << std::endl;
	if(STR_UNKNOWN_CLASS != predict)
	{
            auto cls = std::string("class index ") + std::to_string(top1.first) + " prob. " + std::to_string(top1.second);
	    if(sub == predict)
	    {
	         ++ right;
	    }
	    else
	    {
                 //std::cout << filename <<"  Predicted class index is " << top1.first << " and actual class index is " << index << std::endl;
	         ++ wrong;
	    }
	}
	else
        {
	         ++ wrong;
	}
#else
            //std::cout << "  Predicted class index is " << top1.first << "  class  is " << predict << " real class " << real<< std::endl;
	    if(real == predict)
	    {
	         ++ right;
	    }
	    else
	    {
                 //std::cout << filename <<"  Predicted class index is " << top1.first << " and actual class index is " << index << std::endl;
	         ++ wrong;
	    }
#endif
    }
    std::cout << "right : "  << right << " wrong: " << wrong << "  accuracy : "  << ((float)right/(float)(right+wrong))  << std::endl;
    return 0;
}
int traverse_subdir(ncnn::Net & resnet,Json::Value & labels, std::string sub, int& right ,int & wrong )
{
    // Vector to store the list of filenames
    std::vector<std::string> filenames;
    std::string pattern = VAL_DATA_PATH + sub + "/images/*.JPEG"; 
    //std::string pattern =  sub + "/images/*.JPEG"; 
    if (sub.empty())
    {
	   return -1;
    }
    // Use cv::glob to find all files matching the pattern
    cv::glob(pattern, filenames, false); // 'false' for non-recursive search
     std::cout << "pattern: " << pattern << std::endl;
    // Vector to store the loaded images
    //std::vector<cv::Mat> images;
    size_t count = filenames.size(); // Number of files found

    if (count == 0) {
	    std::cout << "No images found in the directory!" << std::endl;
        return -1;
    }
    std::cout << "total jpeg: "  << count << std::endl;
    // Loop through the filenames and load each image
    for (unsigned int i = 0; i < count; i++) {
	cv::Mat img = cv::imread(filenames[i]); // Read the image
        //std::string filename = get_filename(filenames[i]);
        // Error handling: check if the image loaded successfully
        if (img.empty()) { 
		std::cout << "Error: Could not read image " << filenames[i] << std::endl;
            continue; // Skip to the next iteration
        }
        
        std::vector<float> cls_scores;
        long time = getTimeUsec();
        detect_resnet(resnet,img, cls_scores);
        time = getTimeUsec() - time;
        //printf("detection time: %ld ms\n",time/1000);
        //print_topk(cls_scores, 3);
	auto top1 = topk(cls_scores, 1)[0];
	std::string predict = get_tiny_class(std::to_string(top1.first),labels);
        //std::cout << "  Predicted class index is " << top1.first << "  class  is " << predict << " real class " << sub << std::endl;
	if(STR_UNKNOWN_CLASS != predict)
	{
            auto cls = std::string("class index ") + std::to_string(top1.first) + " prob. " + std::to_string(top1.second);
	    if(sub == predict)
	    {
	         ++ right;
	    }
	    else
	    {
                 //std::cout << filename <<"  Predicted class index is " << top1.first << " and actual class index is " << index << std::endl;
	         ++ wrong;
	    }
	}
	else
        {
	         ++ wrong;
	}
    }
    std::cout << "right : "  << right << " wrong: " << wrong << "  accuracy : "  << ((float)right/(float)(right+wrong))  << std::endl;
    return 0;
}
int main(int argc, char** argv)
{
    // Define the path and file pattern (e.g., all .jpg files in the 'images' folder)
    std::string pattern = "/pytorch/ncnn/build/imagenet-sample-images/*.JPEG"; 
    ncnn::Net resnet;

    resnet.opt.use_vulkan_compute = false;

#if 0
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    resnet.load_param("/pytorch/ncnn/build/onnx2ncnn/model.ncnn.param");
    resnet.load_model("/pytorch/ncnn/build/onnx2ncnn/model.ncnn.bin");
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
#elif 1
    //resnet.load_param("/pytorch/prune/finetune/model.ncnn.param");
    //resnet.load_model("/pytorch/prune/finetune/model.ncnn.bin");
    //resnet.load_param("/pytorch/prune/ncnn/model.ncnn.param");
    //resnet.load_model("/pytorch/prune/ncnn/model.ncnn.bin");
    //resnet.load_param("/pytorch/prune/models/model.ncnn.param");
    //resnet.load_model("/pytorch/prune/models/model.ncnn.bin");
    resnet.load_param("/pytorch/prune/ncnn/resnet50_pruned_v2.ncnn.param");
    resnet.load_model("/pytorch/prune/ncnn/resnet50_pruned_v2.ncnn.bin");
#else
    resnet.load_param("/pytorch/ncnn/build/int8-quant/resnet-int8.param");
    resnet.load_model("/pytorch/ncnn/build/int8-quant/resnet-int8.bin");
#endif
    ncnn::set_omp_num_threads(8);
    // Vector to store the list of filenames
    std::vector<std::string> filenames;
    int right = 0, wrong = 0;
    Json::Value  root;
    if(get_label_json(root))
    {
	    std::cout << "get images label fail!" << std::endl;

	    return -1;
    }
    std::string path = VAL_DATA_PATH;
    try {
        for (const auto & entry : fs::directory_iterator(path)) {
	    if(fs::is_directory(entry.path())) {

	        fs::path relPath = fs::relative(entry.path(), path);
                std::cout << entry.path() << " " << relPath.string() << std::endl;
                //traverse_subdir(root,entry.path());
                v2_traverse_subdir(resnet,root,relPath.string(),right,wrong);
	    }
        }

        //// 2. 递归遍历所有子目录
        //std::cout << "\n--- 递归遍历 ---" << std::endl;
        //for (const auto & entry : fs::recursive_directory_iterator(path)) {
        //    std::cout << entry.path() << std::endl;
        //}
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    std::cout << "right : "  << right << " wrong: " << wrong << "  accuracy : "  << ((float)right/(float)(right+wrong))  << std::endl;
    return 0;
}
