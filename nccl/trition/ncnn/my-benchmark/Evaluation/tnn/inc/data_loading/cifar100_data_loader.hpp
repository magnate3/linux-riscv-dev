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
#include <numeric>
#include <string>
#include <vector>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"

namespace cifar100_constants {
constexpr size_t IMAGE_HEIGHT = 32;
constexpr size_t IMAGE_WIDTH = 32;
constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3;
constexpr size_t NUM_CLASSES = 100;
constexpr size_t NUM_COARSE_CLASSES = 20;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t RECORD_SIZE = 1 + 1 + IMAGE_SIZE;
}  // namespace cifar100_constants

namespace tnn {
/**
 * Enhanced CIFAR-100 data loader for binary format adapted for CNN (2D RGB images)
 * NHWC format: (Batch, Height, Width, Channels)
 */
class CIFAR100DataLoader : public ImageDataLoader {
private:
  std::vector<std::vector<float>> data_;
  std::vector<int> fine_labels_;
  std::vector<int> coarse_labels_;

  std::vector<Tensor> batched_data_;
  std::vector<Tensor> batched_fine_labels_;
  std::vector<Tensor> batched_coarse_labels_;
  bool use_coarse_labels_;
  DType_t dtype_ = DType_t::FP32;

  std::vector<std::string> fine_class_names_ = {
      "apple",       "aquarium_fish", "baby",      "bear",       "beaver",       "bed",
      "bee",         "beetle",        "bicycle",   "bottle",     "bowl",         "boy",
      "bridge",      "bus",           "butterfly", "camel",      "can",          "castle",
      "caterpillar", "cattle",        "chair",     "chimpanzee", "clock",        "cloud",
      "cockroach",   "couch",         "crab",      "crocodile",  "cup",          "dinosaur",
      "dolphin",     "elephant",      "flatfish",  "forest",     "fox",          "girl",
      "hamster",     "house",         "kangaroo",  "keyboard",   "lamp",         "lawn_mower",
      "leopard",     "lion",          "lizard",    "lobster",    "man",          "maple_tree",
      "motorcycle",  "mountain",      "mouse",     "mushroom",   "oak_tree",     "orange",
      "orchid",      "otter",         "palm_tree", "pear",       "pickup_truck", "pine_tree",
      "plain",       "plate",         "poppy",     "porcupine",  "possum",       "rabbit",
      "raccoon",     "ray",           "road",      "rocket",     "rose",         "sea",
      "seal",        "shark",         "shrew",     "skunk",      "skyscraper",   "snail",
      "snake",       "spider",        "squirrel",  "streetcar",  "sunflower",    "sweet_pepper",
      "table",       "tank",          "telephone", "television", "tiger",        "tractor",
      "train",       "trout",         "tulip",     "turtle",     "wardrobe",     "whale",
      "willow_tree", "wolf",          "woman",     "worm"};

  std::vector<std::string> coarse_class_names_ = {"aquatic_mammals",
                                                  "fish",
                                                  "flowers",
                                                  "food_containers",
                                                  "fruit_and_vegetables",
                                                  "household_electrical_devices",
                                                  "household_furniture",
                                                  "insects",
                                                  "large_carnivores",
                                                  "large_man-made_outdoor_things",
                                                  "large_natural_outdoor_scenes",
                                                  "large_omnivores_and_herbivores",
                                                  "medium_mammals",
                                                  "non-insect_invertebrates",
                                                  "people",
                                                  "reptiles",
                                                  "small_mammals",
                                                  "trees",
                                                  "vehicles_1",
                                                  "vehicles_2"};

  template <typename T>
  bool load_multiple_files_impl(const std::vector<std::string> &filenames) {
    data_.clear();
    fine_labels_.clear();
    coarse_labels_.clear();

    for (const auto &filename : filenames) {
      std::ifstream file(filename, std::ios::binary);
      if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
      }

      char buffer[cifar100_constants::RECORD_SIZE];
      size_t records_loaded = 0;

      while (file.read(buffer, cifar100_constants::RECORD_SIZE)) {
        coarse_labels_.push_back(static_cast<int>(static_cast<unsigned char>(buffer[0])));
        fine_labels_.push_back(static_cast<int>(static_cast<unsigned char>(buffer[1])));

        std::vector<float> image_data;
        image_data.reserve(cifar100_constants::IMAGE_SIZE);
        for (size_t i = 2; i < cifar100_constants::RECORD_SIZE; ++i) {
          image_data.push_back(static_cast<float>(static_cast<unsigned char>(buffer[i]) /
                                                  cifar100_constants::NORMALIZATION_FACTOR));
        }

        data_.push_back(std::move(image_data));
        records_loaded++;
      }

      std::cout << "Loaded " << records_loaded << " samples from " << filename << std::endl;
    }

    this->current_index_ = 0;
    std::cout << "Total loaded: " << data_.size() << " samples" << std::endl;
    std::cout << "Using " << (use_coarse_labels_ ? "coarse" : "fine") << " labels" << std::endl;
    return !data_.empty();
  }

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= data_.size()) {
      return false;
    }

    const size_t actual_batch_size = std::min(batch_size, data_.size() - this->current_index_);

    // NHWC format: (Batch, Height, Width, Channels)
    batch_data =
        Tensor::create<T>({actual_batch_size, cifar100_constants::IMAGE_HEIGHT,
                           cifar100_constants::IMAGE_WIDTH, cifar100_constants::NUM_CHANNELS});

    const size_t num_classes = use_coarse_labels_ ? cifar100_constants::NUM_COARSE_CLASSES
                                                  : cifar100_constants::NUM_CLASSES;
    batch_labels = Tensor::create<T>({actual_batch_size, num_classes});
    batch_labels->fill(0.0);

    for (size_t i = 0; i < actual_batch_size; ++i) {
      const auto &image_data = data_[this->current_index_ + i];

      // Convert from CHW (stored format) to HWC (NHWC tensor format)
      for (size_t c = 0; c < cifar100_constants::NUM_CHANNELS; ++c) {
        for (size_t h = 0; h < cifar100_constants::IMAGE_HEIGHT; ++h) {
          for (size_t w = 0; w < cifar100_constants::IMAGE_WIDTH; ++w) {
            size_t pixel_idx =
                c * cifar100_constants::IMAGE_HEIGHT * cifar100_constants::IMAGE_WIDTH +
                h * cifar100_constants::IMAGE_WIDTH + w;
            // NHWC indexing: (batch, height, width, channel)
            batch_data->at<T>({i, h, w, c}) = static_cast<T>(image_data[pixel_idx]);
          }
        }
      }

      const size_t label = use_coarse_labels_
                               ? static_cast<size_t>(coarse_labels_[this->current_index_ + i])
                               : static_cast<size_t>(fine_labels_[this->current_index_ + i]);
      batch_labels->at<T>({i, label}) = static_cast<T>(1.0);
    }

    this->apply_augmentation(batch_data, batch_labels);

    this->current_index_ += actual_batch_size;
    return true;
  }

public:
  CIFAR100DataLoader(bool use_coarse_labels = false, DType_t dtype = DType_t::FP32)
      : ImageDataLoader(), use_coarse_labels_(use_coarse_labels), dtype_(dtype) {
    data_.reserve(50000);
    fine_labels_.reserve(50000);
    coarse_labels_.reserve(50000);
  }

  virtual ~CIFAR100DataLoader() = default;

  /**
   * Load CIFAR-100 data from binary file(s)
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
   * Load CIFAR-100 data from multiple binary files
   */
  bool load_multiple_files(const std::vector<std::string> &filenames) {
    DISPATCH_ON_DTYPE(dtype_, T, return load_multiple_files_impl<T>(filenames));
  }

  /**
   * Get a specific batch size
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
    std::vector<int> shuffled_fine_labels;
    std::vector<int> shuffled_coarse_labels;

    shuffled_data.reserve(data_.size());
    shuffled_fine_labels.reserve(fine_labels_.size());
    shuffled_coarse_labels.reserve(coarse_labels_.size());

    for (const auto &idx : indices) {
      shuffled_data.emplace_back(std::move(data_[idx]));
      shuffled_fine_labels.emplace_back(fine_labels_[idx]);
      shuffled_coarse_labels.emplace_back(coarse_labels_[idx]);
    }

    data_ = std::move(shuffled_data);
    fine_labels_ = std::move(shuffled_fine_labels);
    coarse_labels_ = std::move(shuffled_coarse_labels);
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
    return {cifar100_constants::IMAGE_HEIGHT, cifar100_constants::IMAGE_WIDTH,
            cifar100_constants::NUM_CHANNELS};
  }

  /**
   * Get number of classes
   */
  int get_num_classes() const override {
    return use_coarse_labels_ ? static_cast<int>(cifar100_constants::NUM_COARSE_CLASSES)
                              : static_cast<int>(cifar100_constants::NUM_CLASSES);
  }

  /**
   * Get class names for CIFAR-100
   */
  std::vector<std::string> get_class_names() const override {
    return use_coarse_labels_ ? coarse_class_names_ : fine_class_names_;
  }

  /**
   * Toggle between fine and coarse labels
   */
  void set_use_coarse_labels(bool use_coarse) { use_coarse_labels_ = use_coarse; }

  /**
   * Get data statistics for debugging
   */
  void print_data_stats() const override {
    if (data_.empty()) {
      std::cout << "No data loaded" << std::endl;
      return;
    }

    const size_t num_classes = use_coarse_labels_ ? cifar100_constants::NUM_COARSE_CLASSES
                                                  : cifar100_constants::NUM_CLASSES;
    const auto &labels = use_coarse_labels_ ? coarse_labels_ : fine_labels_;

    std::vector<int> label_counts(num_classes, 0);
    for (const auto &label : labels) {
      if (label >= 0 && label < static_cast<int>(num_classes)) {
        label_counts[label]++;
      }
    }

    std::cout << "CIFAR-100 Dataset Statistics (NHWC format):" << std::endl;
    std::cout << "Total samples: " << data_.size() << std::endl;
    std::cout << "Using " << (use_coarse_labels_ ? "coarse" : "fine") << " labels" << std::endl;
    std::cout << "Image shape: " << cifar100_constants::IMAGE_HEIGHT << "x"
              << cifar100_constants::IMAGE_WIDTH << "x" << cifar100_constants::NUM_CHANNELS
              << std::endl;
    std::cout << "Number of classes: " << num_classes << std::endl;

    if (!data_.empty()) {
      float min_val = *std::min_element(data_[0].begin(), data_[0].end());
      float max_val = *std::max_element(data_[0].begin(), data_[0].end());
      float sum = std::accumulate(data_[0].begin(), data_[0].end(), 0.0f);
      float mean = sum / static_cast<float>(data_[0].size());

      std::cout << "Pixel value range: [" << min_val << ", " << max_val << "]" << std::endl;
      std::cout << "First image mean pixel value: " << mean << std::endl;
    }
  }

  static void create(const std::string &data_path, CIFAR100DataLoader &train_loader,
                     CIFAR100DataLoader &test_loader) {
    if (!train_loader.load_data(data_path + "/cifar-100-binary/train.bin")) {
      throw std::runtime_error("Failed to load training data!");
    }

    if (!test_loader.load_data(data_path + "/cifar-100-binary/test.bin")) {
      throw std::runtime_error("Failed to load test data!");
    }
  }
};
}  // namespace tnn
