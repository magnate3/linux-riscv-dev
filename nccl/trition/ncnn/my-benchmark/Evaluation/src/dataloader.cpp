#include "dataloader.hpp"

namespace evaluate {
// Transform
Transform::Transform() 
: set_flag(false), m_resize(1.0){
    set_size(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        m_mean[i] = 0;
        m_std[i] = 0;
    }
}
Transform::Transform(const int width, const int height, 
                     const float mean_[3], const float std_[3])
: set_flag(false), m_resize(1.0){
    set_size(width, height);
    set_normalize(mean_, std_);
}

void Transform::set_size(const int width, const int height) {
    m_width = width;
    m_height = height;
}

void Transform::set_normalize(const float mean_[3], const float std_[3]) {
    for (size_t i = 0; i < 3; ++i) {
        m_mean[i] = mean_[i];
        m_std[i] = std_[i];
    }
    set_flag = true;
}

void Transform::normalize(ncnn::Mat& mat) {
    mat.substract_mean_normalize(m_mean, m_std);
}

void Transform::normalize(ncnn::Mat& mat, const float mean_[3], const float std_[3]) {
    set_normalize(mean_, std_);
    normalize(mat);
}

ncnn::Mat Transform::transform(cv::Mat& cv_mat, int target_width, int target_height, float portion) {
    /*
     * Take transformation on given cv_mat 
     * to normalize, crop
     */
    int origin_width = cv_mat.cols, origin_height = cv_mat.rows;
    int temp_width = int(target_width / portion);
    int temp_height = int(target_height / portion);
    cv::Size temp_size = cv::Size(temp_width, temp_height);
    
    int roix = (temp_width - target_width) / 2;
    int roiy = (temp_height - target_height) / 2;
    int roiw = target_width, roih = target_height;

    cv::Mat cropped_mat;
    cv::resize(cv_mat, cropped_mat, temp_size, 0, 0);
    ncnn::Mat ncnn_mat = ncnn::Mat::from_pixels_roi_resize(cropped_mat.data, ncnn::Mat::PIXEL_BGR, 
                            cropped_mat.cols, cropped_mat.rows, roix, roiy, roiw, roih, 
                            target_width, target_height);
    // normalize the pixles
    normalize(ncnn_mat);
    return ncnn_mat;
}

// ImageNetDataLoader

ImageNetDataLoader::ImageNetDataLoader()
: m_transform(new Transform()) {
    is_opened = false;
}
ImageNetDataLoader::ImageNetDataLoader(const std::string& source)
: m_transform(new Transform()) {
    is_opened = false;
    open(source);
}

int ImageNetDataLoader::label_parse(const std::string& file_name) {
    // Parse label from filename
    std::size_t label_begin = file_name.find_last_of('_');
    std::size_t label_end = file_name.find_last_of('.');
    int label = atoi(file_name.substr(label_begin+1, label_end - label_begin).c_str());
    return label;
}

bool ImageNetDataLoader::load_data() {
    if (!isOpened()) { 
        std::cout << "Error: DataLoader is not opened. " << std::endl;
        return false;
    }
    if ((m_dir_ptr = readdir(m_dir)) == NULL) {
        std::cout << "All files loaded. " << std::endl;
        return false;
    }
    
    // for ./ and ../ 
    if(m_dir_ptr->d_name[0] == '.') {
        return load_data();
    }
    
    // Get file name and parse label
    std::string file_name = std::string(m_source) + std::string(m_dir_ptr->d_name);
    int label = label_parse(file_name);
    
    // Get image and check
    cv::Mat cv_mat = cv::imread(file_name.c_str(), cv::IMREAD_COLOR);
    if (!cv_mat.data) {
        std::cout << "Error: File " << file_name << " can not be read. " << std::endl;
        return false;
    }

    // Image transformation
    // TODO: move setting to config file
    ncnn::Mat ncnn_mat = m_transform->transform(cv_mat, 224, 224, 0.875);

    // Load to item
    m_item.reset(new DataItem<ncnn::Mat, int>(ncnn_mat, label));
    return true;
}

typename DataItem<ncnn::Mat, int>::ptr ImageNetDataLoader::item() {
    if (load_data()) {
        //std::cout << "data loaded. " << std::endl;
        return m_item;
    }
    else {
        // return nullptr
        m_item.reset();
        return m_item;
    }
}

void ImageNetDataLoader::set_transform(const int height, const int width, 
                               const float mean_[3], const float std_[3]) {
    m_transform->set_size(width, height);
    m_transform->set_normalize(mean_, std_);
}

} // end of namespace