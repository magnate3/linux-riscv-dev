#ifndef __DATALOADER_H__
#define __DATALOADER_H__

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/noncopyable.hpp>
#include <dirent.h>
// ncnn 
#include "mat.h"
#include "layer/crop.h"
// ours
#include "utils.hpp"

namespace evaluate {

class Transform {
public:
    typedef std::shared_ptr<Transform> ptr;
    Transform();
    Transform(const int width, const int height, 
              const float mean_[3], const float std_[3]);
    // Settings
    bool isSet() { return set_flag; }
    void set_size(const int width, const int height);
    void set_normalize(const float mean_[3], const float std_[3]);
    
    // Operations
    void normalize(ncnn::Mat& mat);
    void normalize(ncnn::Mat& mat, const float mean_[3], const float std_[3]);
    ncnn::Mat transform(cv::Mat& cv_mat, int target_width, int target_height, float portion=0.875);

private:
    bool set_flag;
    float m_resize;
    int m_width;
    int m_height;

    float m_mean[3];
    float m_std[3];
};

template<class DATA_TYPE, class LABEL_TYPE>
class DataItem {
public:
    typedef std::shared_ptr<DataItem> ptr;
    DataItem(const DATA_TYPE& data, const LABEL_TYPE& label)
    : m_data(data), m_label(label) {
    }
    DATA_TYPE get_data() const {
        return m_data;
    }
    LABEL_TYPE get_label() const {
        return m_label;
    }
private:
    DATA_TYPE m_data;
    LABEL_TYPE m_label;
};

template<class DATA_TYPE, class LABEL_TYPE>
class BaseDataLoader {
public:
    typedef std::shared_ptr<BaseDataLoader> ptr;
    virtual ~BaseDataLoader() {}
    virtual void open(const std::string& source) {
        m_source = source;
        m_dir = opendir(m_source.c_str());
        is_opened = true;
    }
    virtual typename DataItem<DATA_TYPE, LABEL_TYPE>::ptr item() = 0;
    virtual bool load_data() = 0;
    virtual void set_transform(const int height, const int width, const float mean[3], const float std[3]) = 0;

    bool isOpened() const { return is_opened; }
protected:
    DIR* m_dir;
    struct dirent* m_dir_ptr; // for read file in dataset path
    bool is_opened; // return true if dataset is opened
    std::string m_source; // dataset source path
};

// Dataloader for ImageNet 
class ImageNetDataLoader : BaseDataLoader<ncnn::Mat, int> {
public:
    ImageNetDataLoader();
    ImageNetDataLoader(const std::string& source);
    
    bool load_data() override;
    DataItem<ncnn::Mat, int>::ptr item() override;
    int label_parse(const std::string& file_name);
    void set_transform(const int height, const int width, const float mean[3], const float std[3]) override;
private:
    DataItem<ncnn::Mat, int>::ptr m_item;
    Transform::ptr m_transform;
};

} // end of namespace

#endif