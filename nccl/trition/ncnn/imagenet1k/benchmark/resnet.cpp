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

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>// putText()
#include <stdio.h>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#define CV_FONT_HERSHEY_SIMPLEX         0
#include <iostream>
#include <string>
#include <fstream>
#include <regex>
#include <json/json.h>


int config_to_json() {
  std::ifstream infile;
  infile.open("synset.txt");
  std::vector<std::string> labels;
  std::string line;
  while (getline(infile, line))
  {
    labels.push_back(line);
  }
}
void readFileJson()
{
	Json::Reader reader;
	Json::Value root;

	//从文件中读取，保证当前文件有demo.json文件
	std::ifstream in("class_to_index.json", std::ios::binary);

	if (!in.is_open())
	{
		std::cout << "Error opening file\n";
		return;
	}

	if (reader.parse(in, root))
	{

		if(!root["n02395406"].isNull())
		std::cout << "key<label code> n02395406  vaule<class index> "<< root["n02395406"].asInt() << std::endl;
	}
	else
	{
		std::cout << "parse error\n" << std::endl;
	}

	in.close();
}
std::string get_label() {
    std::string text = "n02395406_hog.JPEG";
    std::regex pattern(R"(^([^_]+)_([^.]+)\.JPEG$)"); // 定义分组
    std::smatch matches;

    // 类似 re.match(pattern, text)
    if (std::regex_match(text, matches, pattern)) {
        std::cout << "Full match: " << matches[0] << std::endl;

        std::cout << text << "  label code: " << matches[1] << std::endl;
	readFileJson();
	return matches[1].str(); 
    } else {
        std::cout << "Match failed" << std::endl;
    }

    return std::string("");
}
long getTimeUsec()
{

    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
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
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

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

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    std::vector<std::string> labels;
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    load_labels("/pytorch/ncnn/build/onnx2ncnn/test/synset_words.txt", labels);
    std::vector<float> cls_scores;
    long time = getTimeUsec();
    detect_resnet(m, cls_scores);
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);

    print_topk(cls_scores, 3);
    get_label();
#if 0   
    for (unsigned int i = 0; i < cls_scores.size(); i++)
    {
	    std::cout << labels[cls_scores[i]] << std::endl;
    }
    for(unsigned int i = 0; i < cls_scores.size(); i++)
   {
     cv::putText(m, labels[cls_scores[i]], cv::Point(50, 50+30*i), CV_FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 100, 200), 2, 8);
   }

   //cv::imshow("result", m);
   cv::imwrite("test_result.jpg", m);
   //cv::waitKey(0);
#endif
    return 0;
}
