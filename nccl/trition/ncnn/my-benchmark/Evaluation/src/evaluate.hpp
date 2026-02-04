#ifndef __EVALUATE__ 
#define __EVALUATE__ 

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <boost/noncopyable.hpp>
// ncnn
#include "net.h"
// ours
#include "dataloader.hpp"

namespace evaluate {
// Dataset path: /mnt/data/dataset/imagenet/
class Network : boost::noncopyable {
public:
    typedef std::shared_ptr<Network> ptr;
    Network(const std::string& net_name);
    void init();
    void process(const ncnn::Mat& image);
    std::vector<int> topk(int k=5);
    std::vector<int> top5(); 
    bool isLoaded() { return is_loaded; }
private:
    std::string m_net_name;
    std::shared_ptr<ncnn::Net> m_net;
    std::shared_ptr<ncnn::Extractor> m_ex;
    std::vector<std::pair<float, int> > cls_scores;
    bool is_loaded;
};

class Evaluate : boost::noncopyable {
public:
    Evaluate(const std::string& source_dir, const std::string& net_name);
    void init();
    void process();
    void accumulate(const std::vector<int>& result, int);
    float top1_accuracy() { return float(top1_accumulate) / float(m_count); }
    float top5_accuracy() { return float(top5_accumulate) / float(m_count); }
private:
    Network::ptr m_network;
    std::shared_ptr<ImageNetDataLoader> m_dataloader;
    int top1_accumulate;
    int top5_accumulate;
    int m_count;
};
} // end of namespace
#endif
