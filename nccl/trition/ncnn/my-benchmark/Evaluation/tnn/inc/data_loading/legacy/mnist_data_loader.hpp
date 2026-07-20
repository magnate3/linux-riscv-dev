/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"

namespace mnist_constants {
constexpr size_t IMAGE_HEIGHT = 28;
constexpr size_t IMAGE_WIDTH = 28;
constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
constexpr size_t NUM_CLASSES = 10;
constexpr size_t NUM_CHANNELS = 1;
constexpr float NORMALIZATION_FACTOR = 255.0f;
}  // namespace mnist_constants

namespace tnn {
namespace legacy {
/**
 * Enhanced MNIST data loader for CSV format adapted for CNN (2D images)
 * Extends ImageDataLoader for proper inheritance
 * Works with type-erased Tensors
 */
class MNISTDataLoader : public ImageDataLoader {
private:
  std::vector<std::vector<float>> data_;
  std::vector<int> labels_;

  std::vector<Tensor> batched_data_;
  std::vector<Tensor> batched_labels_;
  DType_t dtype_ = DType_t::FP32;

  template <typename T>
  bool load_data_impl(const std::string &source) {
    std::ifstream file{source};
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << source << std::endl;
      return false;
    }

    std::string line;
    line.reserve(3136);

    if (!std::getline(file, line)) {
      std::cerr << "Error: Empty file " << source << std::endl;
      return false;
    }

    data_.clear();
    labels_.clear();

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string cell;

      if (!std::getline(ss, cell, ',')) continue;
      labels_.push_back(std::stoi(cell));

      std::vector<float> row;
      row.reserve(mnist_constants::IMAGE_SIZE);

      while (std::getline(ss, cell, ',')) {
        row.push_back(static_cast<float>(std::stod(cell) / mnist_constants::NORMALIZATION_FACTOR));
      }

      if (row.size() != mnist_constants::IMAGE_SIZE) {
        std::cerr << "Warning: Invalid image size " << row.size() << " expected "
                  << mnist_constants::IMAGE_SIZE << std::endl;
        continue;
      }

      data_.push_back(std::move(row));
    }

    this->current_index_ = 0;
    std::cout << "Loaded " << data_.size() << " samples from " << source << std::endl;
    return !data_.empty();
  }

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= data_.size()) {
      return false;
    }

    const size_t actual_batch_size = std::min(batch_size, data_.size() - this->current_index_);

    batch_data = Tensor::create<T>({actual_batch_size, mnist_constants::NUM_CHANNELS,
                                    mnist_constants::IMAGE_HEIGHT, mnist_constants::IMAGE_WIDTH});

    batch_labels = Tensor::create<T>({actual_batch_size, mnist_constants::NUM_CLASSES, 1UL, 1UL});
    batch_labels->fill(0.0);

    for (size_t i = 0; i < actual_batch_size; ++i) {
      const auto &image_data = data_[this->current_index_ + i];

      for (size_t j = 0; j < mnist_constants::IMAGE_SIZE; ++j) {
        batch_data->at<T>({i, 0, j / mnist_constants::IMAGE_WIDTH,
                           j % mnist_constants::IMAGE_WIDTH}) = static_cast<T>(image_data[j]);
      }

      const size_t label = labels_[this->current_index_ + i];
      if (label >= 0 && label < static_cast<int>(mnist_constants::NUM_CLASSES)) {
        batch_labels->at<T>({i, label, 0, 0}) = static_cast<T>(1.0);
      }
    }

    this->apply_augmentation(batch_data, batch_labels);

    this->current_index_ += actual_batch_size;
    return true;
  }

public:
  MNISTDataLoader(DType_t dtype = DType_t::FP32) : ImageDataLoader(), dtype_(dtype) {
    data_.reserve(60000);
    labels_.reserve(60000);
  }

  virtual ~MNISTDataLoader() = default;

  /**
   * Load MNIST data from CSV file
   * @param source Path to CSV file (train.csv or test.csv)
   * @return true if successful, false otherwise
   */
  bool load_data(const std::string &source) override {
    DISPATCH_ON_DTYPE(dtype_, T, return load_data_impl<T>(source));
  }

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
   * Get image dimensions (channels, height, width)
   */
  std::vector<size_t> get_data_shape() const override {
    return {mnist_constants::NUM_CHANNELS, mnist_constants::IMAGE_HEIGHT,
            mnist_constants::IMAGE_WIDTH};
  }

  /**
   * Get number of classes
   */
  int get_num_classes() const override { return static_cast<int>(mnist_constants::NUM_CLASSES); }

  /**
   * Get class names for MNIST (digits 0-9)
   */
  std::vector<std::string> get_class_names() const override {
    std::vector<std::string> names;
    names.reserve(mnist_constants::NUM_CLASSES);
    for (int i = 0; i < static_cast<int>(mnist_constants::NUM_CLASSES); ++i) {
      names.push_back("digit_" + std::to_string(i));
    }
    return names;
  }

  static void create(std::string data_path, MNISTDataLoader &train_loader,
                     MNISTDataLoader &test_loader) {
    if (!train_loader.load_data(data_path + "/mnist/train.csv")) {
      throw std::runtime_error("Failed to load training data!");
    }

    if (!test_loader.load_data(data_path + "/mnist/test.csv")) {
      throw std::runtime_error("Failed to load test data!");
    }
  }
};
}  // namespace legacy
}  // namespace tnn
