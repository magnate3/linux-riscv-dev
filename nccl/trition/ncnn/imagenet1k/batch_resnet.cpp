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
#include "LoadData.h"
#include "class.h"

#include <assert.h>

#include <algorithm> // std::generate
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>// putText()
#include <stdio.h>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#include <filesystem>
#include <cstdlib>
#include <stdlib.h>

#define CV_FONT_HERSHEY_SIMPLEX         0

const int CIFAR_IMAGE_DEPTH = 3;
const int CIFAR_IMAGE_WIDTH = 32;
const int CIFAR_IMAGE_HEIGHT = 32;
const int CIFAR_IMAGE_AREA = CIFAR_IMAGE_WIDTH * CIFAR_IMAGE_HEIGHT;
const int CIFAR_LABEL_SIZE = 1;
const int CIFAR_IMAGE_SIZE = CIFAR_IMAGE_DEPTH * CIFAR_IMAGE_AREA; // 3072 = 3 * 32 * 32


static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
    int K);
static void print_topk(const std::vector<std::pair<int, float>>& topk);
//static void print_topk(const std::vector<float>& topk);
static const char* lookup(int index);
long getTimeUsec()
{

    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}
template<typename T>
std::vector<unsigned int> argsort(const std::vector<T> &arr)
{
  //types::__assert_type<T>();
  if (arr.empty()) return {};
  const unsigned int _size = arr.size();
  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < _size; ++i) indices.push_back(i);
  std::sort(indices.begin(), indices.end(),
            [&arr](const unsigned int a, const unsigned int b)
            { return arr[a] > arr[b]; });
  return indices;
}

template<typename T>
std::vector<unsigned int> argsort(const T *arr, unsigned int _size)
{
  //types::__assert_type<T>();
  if (_size == 0 || arr == nullptr) return {};
  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < _size; ++i) indices.push_back(i);
  std::sort(indices.begin(), indices.end(),
            [arr](const unsigned int a, const unsigned int b)
            { return arr[a] > arr[b]; });
  return indices;
}
static std::vector<float> softmax_cifar10(float* data, int64_t size) {
    auto output = std::vector<float>(size);
    std::transform(data, data + size, output.begin(), expf);
    auto sum =
        std::accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
    std::transform(output.begin(), output.end(), output.begin(),
        [sum](float v) { return v / sum; });
    return output;
}
template<typename T>
std::vector<float> softmax(const T *logits, unsigned int _size, unsigned int &max_id)
{
  //types::__assert_type<T>();
  if (_size == 0 || logits == nullptr) return {};
  float max_prob = 0.f, total_exp = 0.f;
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = std::exp((float) logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > max_prob)
    {
      max_id = i;
      max_prob = softmax_probs[i];
    }
  }
  return softmax_probs;
}
template<typename T> std::vector<float> softmax(
    const std::vector<T> &logits, unsigned int &max_id)
{
  //types::__assert_type<T>();
  if (logits.empty()) return {};
  const unsigned int _size = logits.size();
  float max_prob = 0.f, total_exp = 0.f;
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = std::exp((float) logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > max_prob)
    {
      max_id = i;
      max_prob = softmax_probs[i];
    }
  }
  return softmax_probs;
}
static int detect_resnet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net resnet;

    resnet.opt.use_vulkan_compute = false;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    resnet.load_param("/pytorch/ncnn/build/onnx2ncnn/model.ncnn.param");
    resnet.load_model("/pytorch/ncnn/build/onnx2ncnn/model.ncnn.bin");


                                         
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
                                         
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);
                                         
//const float std_vals[3] = {57.0f, 57.0f, 58.0f};

    //const float std_vals[3] = {0.225f, 0.224f,0.229f};
    //in.substract_mean_normalize(mean_vals, std_vals);
    ncnn::Extractor ex = resnet.create_extractor();

    ex.input("in0", in);

    ncnn::Mat out;
//ex.extract("resnetv17_dense0_fwd", out);

    ex.extract("out0", out);
    std::cout <<"output size: "<< out.total()<< std::endl;
    //ex.extract("resnetv24_stage4_activation8", out);
    cls_scores.resize(out.w);
    unsigned int max_id;
    const unsigned int num_classes = out.w;
    const float *logits = (float *) out.data;
    //std::vector<float> scores = softmax_cifar10(logits, num_classes);
    std::vector<float> scores = softmax<float>(logits, num_classes, max_id);
#if 1
    for (unsigned int j = 0; j < scores.size(); j++)
    {
        cls_scores[j] = scores[j];
    }
#endif

    return 0;
}
#if 0
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
static int load_labels(std::string path, std::vector<std::string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");

    while (!feof(fp))
    {
        char str[1024];
        fgets(str, 1024, fp);
	std::string str_s(str);

        if (str_s.length() > 0)
        {
            for (unsigned int i = 0; i < str_s.length(); i++)
            {
                if (str_s[i] == ' ')
                {
		    std::string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    return 0;
}
#endif
vector<pair<cv::Mat, int>> ReadFirstTenCIFAR10Images(const std::string& filename)
{
    vector<pair<cv::Mat, int>> labeled_images;
    vector<cv::Mat> images;
    ifstream file(filename, std::ios::binary);
    int count = 0;
    vector<int> labels;
    if (file.is_open()) {
        while (!file.eof() && count < 10) {
            unsigned char label;
            unsigned char data[CIFAR_IMAGE_SIZE];
            if (!file.read(reinterpret_cast<char*>(&label), CIFAR_LABEL_SIZE)) {
                break;
            }
            labels.push_back(label);
            if (!file.read(reinterpret_cast<char*>(data), CIFAR_IMAGE_SIZE)) {
                std::cerr << "Error reading image data." << std::endl;
                break;
            }
            cv::Mat channels[3];
            for (int i = 0; i < 3; ++i) {
                channels[i] = cv::Mat(CIFAR_IMAGE_HEIGHT, CIFAR_IMAGE_WIDTH, CV_8UC1, &data[i * CIFAR_IMAGE_AREA]);
            }

            // Merge the separate channels into a single BGR image
            cv::Mat img;
            cv::merge(channels, 3, img);
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

            labeled_images.emplace_back(img, static_cast<int>(label));
            count += 1;
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open the file: " << filename << std::endl;
    }
    return labeled_images;
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

static void print_topk(const std::vector<std::pair<int, float>>& topk) {
    for (const auto& v : topk) {
        std::cout << std::setiosflags(std::ios::left) << std::setw(11)
            << "score[" + std::to_string(v.first) + "]"
            << " =  " << std::setw(12) << v.second
            << " text: " << lookup(v.first)
            << std::resetiosflags(std::ios::left) << std::endl;
    }
}
uint8_t images[NUM_IMAGES][IMAGE_SIZE];
uint8_t labels[NUM_IMAGES];
static const char* lookup(int index) {
    static const char* table[] = {
  #include "cifar_word_list.inc"
    };

    if (index < 0) {
        return "";
    }
    else {
        return table[index];
    }
}
int main(int argc, char** argv)
{
    // Load CIFAR-10 dataset (only training sets)
#if 0
    const string filePaths[] = {
        "../cifar-10-batches-bin/data_batch_1.bin",
        "../cifar-10-batches-bin/data_batch_2.bin",
        "../cifar-10-batches-bin/data_batch_3.bin",
        "../cifar-10-batches-bin/data_batch_4.bin",
        "../cifar-10-batches-bin/data_batch_5.bin"
    };
    const int numFiles = 5;
#elif 1
    const string data_dir = "../cifar-10-batches-bin/test_batch.bin";
    const string output_folder = "images/";
    std::filesystem::create_directory(output_folder);
    auto labeled_images = ReadFirstTenCIFAR10Images(data_dir);
    vector<pair<string, string>> results;
    for (unsigned int i = 0; i < labeled_images.size(); ++i) {
        string output_path = output_folder + "cifar_image_" + std::to_string(i) + ".png";
        cv::imwrite(output_path, labeled_images[i].first);
    }
     for (unsigned int i = 0; i < labeled_images.size(); i++)
    {
	unsigned int top_k =5;
	int lab = labeled_images[i].second;
        const std::string imagepath = "./images/cifar_image_" + std::to_string(i) + ".png";
        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
	    continue;
        }
        std::vector<float> cls_scores;
        long time = getTimeUsec();
        detect_resnet(m, cls_scores);
        time = getTimeUsec() - time;
        printf("detection time: %ld ms\n",time/1000);
#if 1
	std::vector<unsigned int> sorted_indices = argsort<float>(cls_scores);
        printf("tb_top5 size %u \n",sorted_indices.size());
	for (unsigned int i = 0; i < top_k; ++i)
        {
            //content.labels.push_back(sorted_indices[i]);
            //content.scores.push_back(scores[sorted_indices[i]]);
            //content.texts.push_back(class_names[sorted_indices[i]]);
           auto cls = std::string("") + class_names[sorted_indices[i]] + " prob. " + std::to_string(cls_scores[sorted_indices[i]]);
	   cout << "cls: "  << cls << endl;
        }
        string predicted =class_names[sorted_indices[i]];
	results.push_back(std::make_pair(predicted, lookup(lab)));
#else
        //print_topk(cls_scores, 5);
	auto tb_top5 = topk(cls_scores, 5);
	for (unsigned int i = 0; i < top_k; ++i)
        {
            //content.labels.push_back(sorted_indices[i]);
            //content.scores.push_back(scores[sorted_indices[i]]);
            //content.texts.push_back(class_names[sorted_indices[i]]);
	   auto tb_top = tb_top5[i];
           auto cls = std::string("") + std::to_string(tb_top.first) + " prob. " + std::to_string(tb_top.second);
	   cout << "cls: "  << cls << endl;
        }
        //printf("tb_top5 size %u \n",tb_top5.size());
	//auto top1 = tb_top5[0];
        //auto cls = std::string("") + std::to_string(top1.first) + " prob. " + std::to_string(top1.second);
        ////auto cls = std::string("") + lookup(top1.first) + " prob. " + std::to_string(top1.second);
	//cout << "cls: "  << cls << endl;
        //string predicted =lookup(top1.first);
	//results.push_back(std::make_pair(predicted, lookup(lab)));
#endif
    } 
    cout << "Final results:" << endl;
    for (auto n = 0; n < results.size(); n++)
    {
        cout << "Predicted label is " << results[n].first << " and actual label is " << results[n].second << endl;
    }
#else
    const string filePaths[] = {
        "../cifar-10-batches-bin/test_batch.bin",
    };
    const int numFiles = 1;
    loadCIFAR10(filePaths, numFiles, images, labels);
#endif
    return 0;
}
