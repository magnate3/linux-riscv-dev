/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"

// Forward declare stb_image functions
extern "C" {
unsigned char *stbi_load(const char *filename, int *x, int *y, int *channels_in_file,
                         int desired_channels);
void stbi_image_free(void *retval_from_stbi_load);
}

namespace tiny_imagenet_constants {
constexpr size_t IMAGE_HEIGHT = 64;
constexpr size_t IMAGE_WIDTH = 64;
constexpr size_t NUM_CLASSES = 200;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t TRAIN_IMAGES_PER_CLASS = 500;
constexpr size_t VAL_IMAGES = 10000;
constexpr size_t IMAGE_SIZE = NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;
}  // namespace tiny_imagenet_constants

namespace tnn {
/**
 * Tiny ImageNet-200 data loader for JPEG format adapted for CNN (2D RGB images)
 * NHWC format: (Batch, Height, Width, Channels)
 *
 * Dataset structure:
 * - 200 classes
 * - 500 training images per class (100,000 total)
 * - 50 validation images per class (10,000 total)
 * - Images are 64x64 RGB
 */
class TinyImageNetDataLoader : public ImageDataLoader {
private:
  std::vector<float> data_;
  std::vector<int> labels_;

  std::vector<Tensor> batched_data_;
  std::vector<Tensor> batched_labels_;
  DType_t dtype_ = DType_t::FP32;

  std::vector<std::string> class_ids_;                   // WordNet IDs (wnids)
  std::map<std::string, int> class_id_to_index_;         // Map wnid to class index
  std::map<std::string, std::string> class_id_to_name_;  // Map wnid to human-readable name

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    // Otherwise, create batch on demand
    const size_t num_samples = labels_.size();
    if (this->current_index_ >= num_samples) {
      return false;
    }

    const size_t actual_batch_size = std::min(batch_size, num_samples - this->current_index_);

    // NHWC format: (Batch, Height, Width, Channels)
    batch_data = Tensor::create<T>({actual_batch_size, tiny_imagenet_constants::IMAGE_HEIGHT,
                                    tiny_imagenet_constants::IMAGE_WIDTH,
                                    tiny_imagenet_constants::NUM_CHANNELS});

    batch_labels =
        Tensor::create<T>({actual_batch_size, tiny_imagenet_constants::NUM_CLASSES, 1, 1});
    batch_labels->fill(0.0);

    // for (size_t i = 0; i < actual_batch_size; ++i) {
    parallel_for<size_t>(0, actual_batch_size, [&](size_t i) {
      const size_t sample_offset = (this->current_index_ + i) * tiny_imagenet_constants::IMAGE_SIZE;

      // Convert from CHW (stored format) to HWC (NHWC tensor format)
      for (size_t c = 0; c < tiny_imagenet_constants::NUM_CHANNELS; ++c) {
        for (size_t h = 0; h < tiny_imagenet_constants::IMAGE_HEIGHT; ++h) {
          for (size_t w = 0; w < tiny_imagenet_constants::IMAGE_WIDTH; ++w) {
            size_t pixel_idx =
                c * tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH +
                h * tiny_imagenet_constants::IMAGE_WIDTH + w;
            // NHWC indexing: (batch, height, width, channel)
            batch_data->at<T>({i, h, w, c}) = static_cast<T>(data_[sample_offset + pixel_idx]);
          }
        }
      }

      // Set one-hot label
      const size_t label = labels_[this->current_index_ + i];
      if (label >= 0 && label < static_cast<int>(tiny_imagenet_constants::NUM_CLASSES)) {
        batch_labels->at<T>({i, label, 0, 0}) = static_cast<T>(1.0);
      }
    });

    this->apply_augmentation(batch_data, batch_labels);

    this->current_index_ += actual_batch_size;
    return true;
  }

  /**
   * Load class IDs from wnids.txt
   */
  bool load_class_ids(const std::string &dataset_dir) {
    std::string wnids_file = dataset_dir + "/wnids.txt";
    std::ifstream file(wnids_file);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open " << wnids_file << std::endl;
      return false;
    }

    class_ids_.clear();
    class_id_to_index_.clear();

    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
      // Remove trailing whitespace
      line.erase(line.find_last_not_of(" \n\r\t") + 1);
      if (!line.empty()) {
        class_ids_.push_back(line);
        class_id_to_index_[line] = index++;
      }
    }

    std::cout << "Loaded " << class_ids_.size() << " class IDs" << std::endl;
    return class_ids_.size() == tiny_imagenet_constants::NUM_CLASSES;
  }

  /**
   * Load class names from words.txt
   */
  void load_class_names(const std::string &dataset_dir) {
    std::string words_file = dataset_dir + "/words.txt";
    std::ifstream file(words_file);
    if (!file.is_open()) {
      std::cerr << "Warning: Could not open " << words_file << std::endl;
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string wnid, name;
      if (iss >> wnid) {
        // Only load names for the 200 classes we're actually using
        if (class_id_to_index_.find(wnid) != class_id_to_index_.end()) {
          // Get the rest of the line as the class name
          std::getline(iss, name);
          // Remove leading whitespace
          name.erase(0, name.find_first_not_of(" \t"));
          class_id_to_name_[wnid] = name;
        }
      }
    }

    std::cout << "Loaded " << class_id_to_name_.size() << " class names" << std::endl;
  }

  /**
   * Load a JPEG image and convert to normalized float data
   */
  bool load_jpeg_image(const std::string &image_path, float *image_data_ptr) {
    int width, height, channels;
    unsigned char *img = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

    if (!img) {
      std::cerr << "Error loading image: " << image_path << std::endl;
      return false;
    }

    if (width != static_cast<int>(tiny_imagenet_constants::IMAGE_WIDTH) ||
        height != static_cast<int>(tiny_imagenet_constants::IMAGE_HEIGHT)) {
      std::cerr << "Warning: Image " << image_path << " has unexpected dimensions: " << width << "x"
                << height << std::endl;
      stbi_image_free(img);
      return false;
    }

    // Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) format
    // and normalize to [0, 1]
    for (size_t c = 0; c < tiny_imagenet_constants::NUM_CHANNELS; ++c) {
      for (size_t h = 0; h < tiny_imagenet_constants::IMAGE_HEIGHT; ++h) {
        for (size_t w = 0; w < tiny_imagenet_constants::IMAGE_WIDTH; ++w) {
          size_t src_idx = (h * tiny_imagenet_constants::IMAGE_WIDTH + w) * 3 + c;
          size_t dst_idx =
              c * tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH +
              h * tiny_imagenet_constants::IMAGE_WIDTH + w;
          image_data_ptr[dst_idx] = img[src_idx] / tiny_imagenet_constants::NORMALIZATION_FACTOR;
        }
      }
    }

    stbi_image_free(img);
    return true;
  }

  /**
   * Load training data from directory structure (parallelized)
   */
  bool load_train_data(const std::string &dataset_dir) {
    std::string train_dir = dataset_dir + "/train";

    if (!std::filesystem::exists(train_dir)) {
      std::cerr << "Error: Training directory not found: " << train_dir << std::endl;
      return false;
    }

    std::vector<std::pair<std::string, int>> image_paths;
    image_paths.reserve(tiny_imagenet_constants::NUM_CLASSES *
                        tiny_imagenet_constants::TRAIN_IMAGES_PER_CLASS);

    for (const auto &class_id : class_ids_) {
      std::string class_dir = train_dir + "/" + class_id + "/images";

      if (!std::filesystem::exists(class_dir)) {
        std::cerr << "Warning: Class directory not found: " << class_dir << std::endl;
        continue;
      }

      int class_index = class_id_to_index_[class_id];

      for (const auto &entry : std::filesystem::directory_iterator(class_dir)) {
        if (entry.path().extension() == ".JPEG") {
          image_paths.emplace_back(entry.path().string(), class_index);
        }
      }
    }

    const size_t num_images = image_paths.size();
    std::cout << "Found " << num_images << " images to load..." << std::endl;

    data_.resize(num_images * tiny_imagenet_constants::IMAGE_SIZE);
    labels_.resize(num_images);
    std::vector<bool> load_success(num_images, false);

    parallel_for<size_t>(0, num_images, [&](size_t i) {
      const auto &[path, class_index] = image_paths[i];
      if (load_jpeg_image(path, &data_[i * tiny_imagenet_constants::IMAGE_SIZE])) {
        labels_[i] = class_index;
        load_success[i] = true;
      }
    });

    size_t total_loaded = std::count(load_success.begin(), load_success.end(), true);

    if (total_loaded < num_images) {
      size_t write_idx = 0;
      for (size_t i = 0; i < num_images; ++i) {
        if (load_success[i]) {
          if (write_idx != i) {
            std::copy(data_.begin() + i * tiny_imagenet_constants::IMAGE_SIZE,
                      data_.begin() + (i + 1) * tiny_imagenet_constants::IMAGE_SIZE,
                      data_.begin() + write_idx * tiny_imagenet_constants::IMAGE_SIZE);
            labels_[write_idx] = labels_[i];
          }
          ++write_idx;
        }
      }
      data_.resize(total_loaded * tiny_imagenet_constants::IMAGE_SIZE);
      labels_.resize(total_loaded);
    }

    std::cout << "Loaded " << total_loaded << " training images" << std::endl;
    return total_loaded > 0;
  }

  /**
   * Load validation data from annotations file
   */
  bool load_val_data(const std::string &dataset_dir) {
    std::string val_dir = dataset_dir + "/val";
    std::string val_annotations = val_dir + "/val_annotations.txt";
    std::string val_images_dir = val_dir + "/images";

    if (!std::filesystem::exists(val_annotations)) {
      std::cerr << "Error: Validation annotations not found: " << val_annotations << std::endl;
      return false;
    }

    std::ifstream file(val_annotations);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open validation annotations file" << std::endl;
      return false;
    }

    std::string line;
    size_t total_loaded = 0;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string image_name, class_id;

      // Format: image_name class_id x y w h
      if (iss >> image_name >> class_id) {
        std::string image_path = val_images_dir + "/" + image_name;

        if (class_id_to_index_.find(class_id) == class_id_to_index_.end()) {
          std::cerr << "Warning: Unknown class ID: " << class_id << std::endl;
          continue;
        }

        int class_index = class_id_to_index_[class_id];

        // Prepare space in flat vector
        size_t current_offset = data_.size();
        data_.resize(current_offset + tiny_imagenet_constants::IMAGE_SIZE);

        if (load_jpeg_image(image_path, &data_[current_offset])) {
          labels_.push_back(class_index);
          total_loaded++;
        } else {
          // If load failed, revert resize
          data_.resize(current_offset);
        }
      }
    }

    std::cout << "Loaded " << total_loaded << " validation images" << std::endl;
    return total_loaded > 0;
  }

public:
  TinyImageNetDataLoader(DType_t dtype = DType_t::FP32) : ImageDataLoader(), dtype_(dtype) {
    data_.reserve(100000 * tiny_imagenet_constants::IMAGE_SIZE);  // Reserve for training set
    labels_.reserve(100000);
  }

  virtual ~TinyImageNetDataLoader() = default;

  /**
   * Load Tiny ImageNet-200 data
   * @param source Path to dataset directory containing train/, val/, wnids.txt, and words.txt
   * @param is_train If true, load training data; if false, load validation data
   * @return true if successful, false otherwise
   */
  bool load_data(const std::string &source, bool is_train = true) {
    data_.clear();
    labels_.clear();

    // Load class IDs and names
    if (!load_class_ids(source)) {
      return false;
    }

    load_class_names(source);

    // Load the appropriate dataset
    bool success;
    if (is_train) {
      success = load_train_data(source);
    } else {
      success = load_val_data(source);
    }

    if (success) {
      this->current_index_ = 0;
      std::cout << "Total loaded: " << labels_.size() << " samples" << std::endl;
    }

    return success;
  }

  /**
   * Overload for compatibility with BaseDataLoader interface
   */
  bool load_data(const std::string &source) override {
    return load_data(source, true);  // Default to training data
  }

  /**
   * Get a specific batch size (supports both pre-computed and on-demand batches)
   */
  bool get_batch(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) override {
    DISPATCH_ON_DTYPE(dtype_, T, return get_batch_impl<T>(batch_size, batch_data, batch_labels));
  }

  /**
   * Reset iterator to beginning of dataset
   */
  void reset() override { this->current_index_ = 0; }

  /**
   * Shuffle the dataset
   */
  void shuffle() override {
    // Shuffle raw data
    if (labels_.empty()) return;

    const size_t num_samples = labels_.size();
    std::vector<size_t> indices = this->generate_shuffled_indices(num_samples);

    std::vector<float> shuffled_data;
    std::vector<int> shuffled_labels;
    shuffled_data.resize(data_.size());
    shuffled_labels.reserve(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
      size_t idx = indices[i];
      std::copy(data_.begin() + idx * tiny_imagenet_constants::IMAGE_SIZE,
                data_.begin() + (idx + 1) * tiny_imagenet_constants::IMAGE_SIZE,
                shuffled_data.begin() + i * tiny_imagenet_constants::IMAGE_SIZE);
      shuffled_labels.emplace_back(labels_[idx]);
    }

    data_ = std::move(shuffled_data);
    labels_ = std::move(shuffled_labels);
    this->current_index_ = 0;
  }

  /**
   * Get the total number of samples in the dataset
   */
  size_t size() const override { return labels_.size(); }

  /**
   * Get image dimensions (height, width, channels) for NHWC format
   */
  std::vector<size_t> get_data_shape() const override {
    return {tiny_imagenet_constants::IMAGE_HEIGHT, tiny_imagenet_constants::IMAGE_WIDTH,
            tiny_imagenet_constants::NUM_CHANNELS};
  }

  /**
   * Get number of classes
   */
  int get_num_classes() const override {
    return static_cast<int>(tiny_imagenet_constants::NUM_CLASSES);
  }

  /**
   * Get class names for Tiny ImageNet-200
   */
  std::vector<std::string> get_class_names() const override {
    std::vector<std::string> names;
    names.reserve(class_ids_.size());

    for (const auto &class_id : class_ids_) {
      auto it = class_id_to_name_.find(class_id);
      if (it != class_id_to_name_.end()) {
        names.push_back(it->second);
      } else {
        names.push_back(class_id);  // Fall back to wnid if name not found
      }
    }

    return names;
  }

  /**
   * Get class IDs (WordNet IDs)
   */
  std::vector<std::string> get_class_ids() const { return class_ids_; }

  /**
   * Get data statistics for debugging
   */
  void print_data_stats() const override {
    if (labels_.empty()) {
      std::cout << "No data loaded" << std::endl;
      return;
    }

    std::vector<int> label_counts(tiny_imagenet_constants::NUM_CLASSES, 0);
    for (const auto &label : labels_) {
      if (label >= 0 && label < static_cast<int>(tiny_imagenet_constants::NUM_CLASSES)) {
        label_counts[label]++;
      }
    }

    std::cout << "Tiny ImageNet-200 Dataset Statistics (NHWC format):" << std::endl;
    std::cout << "Total samples: " << labels_.size() << std::endl;
    std::cout << "Image shape: " << tiny_imagenet_constants::IMAGE_HEIGHT << "x"
              << tiny_imagenet_constants::IMAGE_WIDTH << "x"
              << tiny_imagenet_constants::NUM_CHANNELS << std::endl;
    std::cout << "Number of classes: " << tiny_imagenet_constants::NUM_CLASSES << std::endl;

    // Show distribution for first 10 classes
    std::cout << "Class distribution (first 10 classes):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(class_ids_.size())); ++i) {
      std::string class_name = class_ids_[i];
      auto it = class_id_to_name_.find(class_name);
      if (it != class_id_to_name_.end()) {
        class_name = it->second;
      }
      std::cout << "  Class " << i << " (" << class_ids_[i] << " - " << class_name
                << "): " << label_counts[i] << " samples" << std::endl;
    }

    if (!data_.empty()) {
      float min_val =
          *std::min_element(data_.begin(), data_.begin() + tiny_imagenet_constants::IMAGE_SIZE);
      float max_val =
          *std::max_element(data_.begin(), data_.begin() + tiny_imagenet_constants::IMAGE_SIZE);
      float sum = std::accumulate(data_.begin(),
                                  data_.begin() + tiny_imagenet_constants::IMAGE_SIZE, float(1.0));
      float mean = static_cast<float>(sum) / tiny_imagenet_constants::IMAGE_SIZE;

      std::cout << "Pixel value range: [" << (float)min_val << ", " << (float)max_val << "]"
                << std::endl;
      std::cout << "First image mean pixel value: " << (float)mean << std::endl;
    }
  }

  static void create(std::string data_path, TinyImageNetDataLoader &train_loader,
                     TinyImageNetDataLoader &val_loader) {
    if (!train_loader.load_data(data_path, true)) {
      throw std::runtime_error("Failed to load training data!");
    }
    if (!val_loader.load_data(data_path, false)) {
      throw std::runtime_error("Failed to load validation data!");
    }
  }
};
}  // namespace tnn
