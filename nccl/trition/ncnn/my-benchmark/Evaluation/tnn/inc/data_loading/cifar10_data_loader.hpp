/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"

namespace cifar10_constants {
constexpr size_t IMAGE_HEIGHT = 32;
constexpr size_t IMAGE_WIDTH = 32;
constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3;
constexpr size_t NUM_CLASSES = 10;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t RECORD_SIZE = 1 + IMAGE_SIZE;
}  // namespace cifar10_constants

namespace tnn {
/**
 *  CIFAR-10 data loader for binary format adapted for CNN (2D RGB images)
 *  NHWC format: (Batch, Height, Width, Channels)
 */
class CIFAR10DataLoader : public ImageDataLoader {
private:
  std::vector<std::vector<float>> data_;
  std::vector<int> labels_;

  DType_t dtype_ = DType_t::FP32;

  std::vector<std::string> class_names_ = {"airplane", "automobile", "bird",  "cat",  "deer",
                                           "dog",      "frog",       "horse", "ship", "truck"};

  template <typename T>
  bool load_multiple_files_impl(const std::vector<std::string> &filenames) {
    data_.clear();
    labels_.clear();

    for (const auto &filename : filenames) {
      std::ifstream file(filename, std::ios::binary);
      if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
      }

      char buffer[cifar10_constants::RECORD_SIZE];
      size_t records_loaded = 0;

      while (file.read(buffer, cifar10_constants::RECORD_SIZE)) {
        labels_.push_back(static_cast<int>(static_cast<unsigned char>(buffer[0])));
        std::vector<float> image_data;
        image_data.reserve(cifar10_constants::IMAGE_SIZE);

        for (size_t i = 1; i < cifar10_constants::RECORD_SIZE; ++i) {
          image_data.push_back(static_cast<float>(static_cast<unsigned char>(buffer[i]) /
                                                  cifar10_constants::NORMALIZATION_FACTOR));
        }

        data_.push_back(std::move(image_data));
        records_loaded++;
      }

      std::cout << "Loaded " << records_loaded << " samples from " << filename << std::endl;
    }

    this->current_index_ = 0;
    std::cout << "Total loaded: " << data_.size() << " samples" << std::endl;
    return !data_.empty();
  }

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= data_.size()) {
      return false;
    }

    const size_t actual_batch_size = std::min(batch_size, data_.size() - this->current_index_);
    const size_t height = cifar10_constants::IMAGE_HEIGHT;
    const size_t width = cifar10_constants::IMAGE_WIDTH;
    const size_t channels = cifar10_constants::NUM_CHANNELS;
    const size_t num_classes = cifar10_constants::NUM_CLASSES;
    // NHWC format: (Batch, Height, Width, Channels)
    batch_data = Tensor::create<T>({actual_batch_size, height, width, channels});

    batch_labels = Tensor::create<T>({actual_batch_size, num_classes, 1, 1});
    batch_labels->fill(0.0);

    T *data = batch_data->data_as<T>();
    T *labels = batch_labels->data_as<T>();

    parallel_for<size_t>(0, actual_batch_size, [&](size_t i) {
      const std::vector<float> &image_data = data_[this->current_index_ + i];
      size_t data_pixel_idx = i * height * width * channels;

      for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t pixel_idx = c * height * width + h * width + w;
            data[data_pixel_idx++] = static_cast<T>(image_data[pixel_idx]);
          }
        }
      }

      const size_t label = labels_[this->current_index_ + i];
      if (label >= 0 && label < static_cast<int>(num_classes)) {
        labels[i * num_classes + label] = static_cast<T>(1.0);
      }
    });

    this->apply_augmentation(batch_data, batch_labels);

    this->current_index_ += actual_batch_size;
    return true;
  }

public:
  CIFAR10DataLoader(DType_t dtype = DType_t::FP32) : ImageDataLoader(), dtype_(dtype) {
    data_.reserve(50000);
    labels_.reserve(50000);
  }

  virtual ~CIFAR10DataLoader() = default;

  /**
   * Load CIFAR-10 data from binary file(s)
   * @param source Path to binary file or directory containing multiple files
   * @return true if successful, false otherwise
   */
  bool load_data(const std::string &source) override {
    std::vector<std::string> filenames;

    if (source.find(".bin") != std::string::npos) {
      filenames.push_back(source);
    } else {
      std::cerr << "Error: For multiple files, use load_multiple_files() method" << std::endl;
      return false;
    }

    return load_multiple_files(filenames);
  }

  /**
   * Load CIFAR-10 data from multiple binary files
   * @param filenames Vector of file paths to load
   * @return true if successful, false otherwise
   */
  bool load_multiple_files(const std::vector<std::string> &filenames) {
    DISPATCH_ON_DTYPE(dtype_, T, return load_multiple_files_impl<T>(filenames));
  }

  /**
   * Get a specific batch size (supports both pre-computed and on-demand
   * batches)
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
    if (data_.empty()) return;

    std::vector<size_t> indices = this->generate_shuffled_indices(data_.size());

    std::vector<std::vector<float>> shuffled_data;
    std::vector<int> shuffled_labels;
    shuffled_data.reserve(data_.size());
    shuffled_labels.reserve(labels_.size());

    for (const auto &idx : indices) {
      shuffled_data.emplace_back(std::move(data_[idx]));
      shuffled_labels.emplace_back(labels_[idx]);
    }

    data_ = std::move(shuffled_data);
    labels_ = std::move(shuffled_labels);
    this->current_index_ = 0;
  }

  /**
   * Get the total number of samples in the dataset
   */
  size_t size() const override { return data_.size(); }

  /**
   * Get image dimensions (height, width, channels) for NHWC format
   */
  std::vector<size_t> get_data_shape() const override {
    return {cifar10_constants::IMAGE_HEIGHT, cifar10_constants::IMAGE_WIDTH,
            cifar10_constants::NUM_CHANNELS};
  }

  /**
   * Get number of classes
   */
  int get_num_classes() const override { return static_cast<int>(cifar10_constants::NUM_CLASSES); }

  /**
   * Get class names for CIFAR-10
   */
  std::vector<std::string> get_class_names() const override { return class_names_; }

  /**
   * Get data statistics for debugging
   */
  void print_data_stats() const override {
    if (data_.empty()) {
      std::cout << "No data loaded" << std::endl;
      return;
    }

    std::vector<int> label_counts(cifar10_constants::NUM_CLASSES, 0);
    for (const auto &label : labels_) {
      if (label >= 0 && label < static_cast<int>(cifar10_constants::NUM_CLASSES)) {
        label_counts[label]++;
      }
    }

    std::cout << "CIFAR-10 Dataset Statistics (NHWC format):" << std::endl;
    std::cout << "Total samples: " << data_.size() << std::endl;
    std::cout << "Image shape: " << cifar10_constants::IMAGE_HEIGHT << "x"
              << cifar10_constants::IMAGE_WIDTH << "x" << cifar10_constants::NUM_CHANNELS
              << std::endl;
    std::cout << "Class distribution:" << std::endl;
    for (int i = 0; i < static_cast<int>(cifar10_constants::NUM_CLASSES); ++i) {
      std::cout << "  " << class_names_[i] << " (" << i << "): " << label_counts[i] << " samples"
                << std::endl;
    }

    if (!data_.empty()) {
      float min_val = *std::min_element(data_[0].begin(), data_[0].end());
      float max_val = *std::max_element(data_[0].begin(), data_[0].end());
      float sum = std::accumulate(data_[0].begin(), data_[0].end(), 0.0f);
      float mean = sum / static_cast<float>(data_[0].size());

      std::cout << "Pixel value range: [" << min_val << ", " << max_val << "]" << std::endl;
      std::cout << "First image mean pixel value: " << mean << std::endl;
    }
  }

  static void create(const std::string &data_path, CIFAR10DataLoader &train_loader,
                     CIFAR10DataLoader &test_loader) {
    if (!train_loader.load_multiple_files({data_path + "/cifar-10-batches-bin/data_batch_1.bin",
                                           data_path + "/cifar-10-batches-bin/data_batch_2.bin",
                                           data_path + "/cifar-10-batches-bin/data_batch_3.bin",
                                           data_path + "/cifar-10-batches-bin/data_batch_4.bin",
                                           data_path + "/cifar-10-batches-bin/data_batch_5.bin"})) {
      throw std::runtime_error("Failed to load training data!");
    }

    if (!test_loader.load_data(data_path + "/cifar-10-batches-bin/test_batch.bin")) {
      throw std::runtime_error("Failed to load test data!");
    }
  }
};
}  // namespace tnn
