/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "data_augmentation/augmentation.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

/**
 * Abstract base class for all data loaders
 * Provides common interface and functionality for training neural networks
 */
class BaseDataLoader {
public:
  virtual ~BaseDataLoader() = default;

  /**
   * Load data from file(s)
   * @param source Data source (file path, directory, etc.)
   * @return true if successful, false otherwise
   */
  virtual bool load_data(const std::string &source) = 0;

  /**
   * Get a specific batch size
   * @param batch_size Number of samples per batch
   * @param batch_data Output tensor for features/input data
   * @param batch_labels Output tensor for labels/targets
   * @return true if batch was retrieved, false if no more data
   */
  virtual bool get_batch(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) = 0;

  /**
   * Reset iterator to beginning of dataset
   */
  virtual void reset() = 0;

  /**
   * Shuffle the dataset
   */
  virtual void shuffle() = 0;

  /**
   * Get the total number of samples in the dataset
   */
  virtual size_t size() const = 0;

  /**
   * Get data shape
   */
  virtual std::vector<size_t> get_data_shape() const = 0;

  /**
   * Print data statistics for debugging
   */
  virtual void print_data_stats() const {}

  /**
   * Set random seed for reproducible shuffling
   */
  virtual void set_seed(unsigned int seed) { rng_.seed(seed); }

  /**
   * Get random number generator for derived classes
   */
  std::mt19937 &get_rng() { return rng_; }
  const std::mt19937 &get_rng() const { return rng_; }

  void set_augmentation(std::unique_ptr<AugmentationStrategy> aug) {
    augmentation_ = std::move(aug);
  }

  /**
   * Remove augmentation strategy (useful for validation/test sets)
   */
  void clear_augmentation() { augmentation_.reset(); }

  /**
   * Check if augmentation is enabled
   */
  bool has_augmentation() const { return augmentation_ != nullptr; }

protected:
  size_t current_index_ = 0;
  mutable std::mt19937 rng_{std::random_device{}()};
  std::unique_ptr<AugmentationStrategy> augmentation_;

  /**
   * Apply augmentation to batch if augmentation strategy is set
   * Called internally by derived classes after loading batch data
   */
  void apply_augmentation(Tensor &batch_data, Tensor &batch_labels) {
    if (augmentation_) {
      augmentation_->apply(batch_data, batch_labels);
    }
  }

  /**
   * Utility function to shuffle indices
   */
  std::vector<size_t> generate_shuffled_indices(size_t data_size) const {
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    return indices;
  }
};

}  // namespace tnn
