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
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "data_loader.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

// needs an overhaul
class WiFiDataLoader : public BaseDataLoader {
private:
  std::vector<std::vector<float>> features_;
  std::vector<std::vector<float>> targets_;

  size_t num_features_;
  size_t num_outputs_;
  bool is_regression_;

  std::vector<float> feature_means_;
  std::vector<float> feature_stds_;
  std::vector<float> target_means_;
  std::vector<float> target_stds_;
  bool is_normalized_;

public:
  WiFiDataLoader(bool is_regression = true)
      : num_features_(0), num_outputs_(0), is_regression_(is_regression), is_normalized_(false) {
    features_.reserve(20000);
    targets_.reserve(20000);
  }

  bool load_data(const std::string &filename) override {
    return load_data(filename, 0, 0, 0, 0, true);
  }

  bool load_data(const std::string &filename, size_t feature_start_col, size_t feature_end_col,
                 size_t target_start_col, size_t target_end_col, bool has_header) {
    std::ifstream file{filename};
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return false;
    }

    features_.clear();
    targets_.clear();

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
      if (first_line && has_header) {
        first_line = false;
        continue;
      }

      std::stringstream ss(line);
      std::string cell;
      std::vector<std::string> row;

      while (std::getline(ss, cell, ',')) {
        row.push_back(cell);
      }

      if (row.empty()) continue;

      if (feature_end_col == 0) {
        if (is_regression_) {
          feature_end_col = row.size() - 2;
          target_start_col = row.size() - 2;
          target_end_col = row.size();
        } else {
          feature_end_col = row.size() - 1;
          target_start_col = row.size() - 1;
          target_end_col = row.size();
        }
      }

      std::vector<float> feature_row;
      for (size_t i = feature_start_col; i < std::min(feature_end_col, row.size()); ++i) {
        try {
          float value = std::stof(row[i]);

          if (value == 100.0f || value == 0.0f) {
            value = -100.0f;
          }
          feature_row.push_back(value);
        } catch (const std::exception &) {
          feature_row.push_back(-100.0f);
        }
      }

      std::vector<float> target_row;
      for (size_t i = target_start_col; i < std::min(target_end_col, row.size()); ++i) {
        try {
          target_row.push_back(std::stof(row[i]));
        } catch (const std::exception &) {
          target_row.push_back(0.0f);
        }
      }

      if (!feature_row.empty() && !target_row.empty()) {
        features_.push_back(std::move(feature_row));
        targets_.push_back(std::move(target_row));
      }
    }

    if (features_.empty()) {
      std::cerr << "Error: No valid data loaded from " << filename << std::endl;
      return false;
    }

    num_features_ = features_[0].size();
    num_outputs_ = targets_[0].size();
    current_index_ = 0;

    std::cout << "Loaded " << features_.size() << " samples from " << filename << std::endl;
    std::cout << "Features: " << num_features_ << ", Outputs: " << num_outputs_ << std::endl;
    std::cout << "Mode: " << (is_regression_ ? "Regression" : "Classification") << std::endl;

    if (is_regression_ && targets_.size() > 0) {
      std::cout << "First 5 target coordinate samples:" << std::endl;
      for (size_t i = 0; i < std::min(size_t(5), targets_.size()); ++i) {
        std::cout << "  Sample " << i << ": (";
        for (size_t j = 0; j < targets_[i].size(); ++j) {
          std::cout << targets_[i][j];
          if (j < targets_[i].size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
      }

      if (num_outputs_ >= 2) {
        float min_x = targets_[0][0], max_x = targets_[0][0];
        float min_y = targets_[0][1], max_y = targets_[0][1];

        for (const auto &target : targets_) {
          min_x = std::min(min_x, target[0]);
          max_x = std::max(max_x, target[0]);
          min_y = std::min(min_y, target[1]);
          max_y = std::max(max_y, target[1]);
        }

        std::cout << "Coordinate ranges:" << std::endl;
        std::cout << "  X: [" << min_x << ", " << max_x << "] (range: " << (max_x - min_x) << ")"
                  << std::endl;
        std::cout << "  Y: [" << min_y << ", " << max_y << "] (range: " << (max_y - min_y) << ")"
                  << std::endl;
      }
    }

    return true;
  }

  void normalize_data() {
    if (features_.empty()) {
      std::cerr << "Warning: No data to normalize!" << std::endl;
      return;
    }

    feature_means_.assign(num_features_, 0.0f);
    feature_stds_.assign(num_features_, 0.0f);

    for (const auto &sample : features_) {
      for (size_t i = 0; i < num_features_; ++i) {
        feature_means_[i] += sample[i];
      }
    }

    const float inv_size = 1.0f / static_cast<float>(features_.size());
    for (size_t i = 0; i < num_features_; ++i) {
      feature_means_[i] *= inv_size;
    }

    for (const auto &sample : features_) {
      for (size_t i = 0; i < num_features_; ++i) {
        const float diff = sample[i] - feature_means_[i];
        feature_stds_[i] += diff * diff;
      }
    }

    for (size_t i = 0; i < num_features_; ++i) {
      feature_stds_[i] = std::sqrt(feature_stds_[i] * inv_size);
      if (feature_stds_[i] < 1e-8f) {
        feature_stds_[i] = 1.0f;
      }
    }

    for (size_t s = 0; s < features_.size(); ++s) {
      for (size_t i = 0; i < num_features_; ++i) {
        features_[s][i] = (features_[s][i] - feature_means_[i]) / feature_stds_[i];
      }
    }

    if (is_regression_) {
      target_means_.assign(num_outputs_, 0.0f);
      target_stds_.assign(num_outputs_, 0.0f);

      for (const auto &sample : targets_) {
        for (size_t i = 0; i < num_outputs_; ++i) {
          target_means_[i] += sample[i];
        }
      }

      for (size_t i = 0; i < num_outputs_; ++i) {
        target_means_[i] *= inv_size;
      }

      std::cout << "Target means: ";
      for (size_t i = 0; i < target_means_.size(); ++i) {
        std::cout << target_means_[i] << " ";
      }
      std::cout << std::endl;

      for (const auto &sample : targets_) {
        for (size_t i = 0; i < num_outputs_; ++i) {
          const float diff = sample[i] - target_means_[i];
          target_stds_[i] += diff * diff;
        }
      }

      for (size_t i = 0; i < num_outputs_; ++i) {
        target_stds_[i] = std::sqrt(target_stds_[i] * inv_size);
        if (target_stds_[i] < 1e-8f) {
          target_stds_[i] = 1.0f;
        }
      }

      std::cout << "Target stds: ";
      for (size_t i = 0; i < target_stds_.size(); ++i) {
        std::cout << target_stds_[i] << " ";
      }
      std::cout << std::endl;
      for (size_t s = 0; s < targets_.size(); ++s) {
        for (size_t i = 0; i < num_outputs_; ++i) {
          targets_[s][i] = (targets_[s][i] - target_means_[i]) / target_stds_[i];
        }
      }
    }

    is_normalized_ = true;
    std::cout << "Data normalization completed!" << std::endl;
  }

  void apply_normalization(const std::vector<float> &feature_means,
                           const std::vector<float> &feature_stds,
                           const std::vector<float> &target_means,
                           const std::vector<float> &target_stds) {
    if (features_.empty()) {
      std::cerr << "Warning: No data to normalize!" << std::endl;
      return;
    }

    if (feature_means.size() != num_features_ || feature_stds.size() != num_features_) {
      std::cerr << "Error: Feature statistics size mismatch!" << std::endl;
      return;
    }

    feature_means_ = feature_means;
    feature_stds_ = feature_stds;

    for (size_t s = 0; s < features_.size(); ++s) {
      for (size_t i = 0; i < num_features_; ++i) {
        features_[s][i] = (features_[s][i] - feature_means_[i]) / feature_stds_[i];
      }
    }

    if (is_regression_ && !target_means.empty() && !target_stds.empty()) {
      if (target_means.size() != num_outputs_ || target_stds.size() != num_outputs_) {
        std::cerr << "Error: Target statistics size mismatch!" << std::endl;
        return;
      }

      target_means_ = target_means;
      target_stds_ = target_stds;

      for (size_t s = 0; s < targets_.size(); ++s) {
        for (size_t i = 0; i < num_outputs_; ++i) {
          targets_[s][i] = (targets_[s][i] - target_means_[i]) / target_stds_[i];
        }
      }
    }

    is_normalized_ = true;
    std::cout << "Data normalization using external statistics completed!" << std::endl;
  }

  std::vector<float> get_feature_means() const { return feature_means_; }
  std::vector<float> get_feature_stds() const { return feature_stds_; }
  std::vector<float> get_target_means() const { return target_means_; }
  std::vector<float> get_target_stds() const { return target_stds_; }

  void shuffle() override {
    if (features_.empty()) return;

    std::vector<size_t> indices = this->generate_shuffled_indices(features_.size());

    std::vector<std::vector<float>> shuffled_features, shuffled_targets;
    shuffled_features.reserve(features_.size());
    shuffled_targets.reserve(targets_.size());

    for (const auto &idx : indices) {
      shuffled_features.emplace_back(std::move(features_[idx]));
      shuffled_targets.emplace_back(std::move(targets_[idx]));
    }

    features_ = std::move(shuffled_features);
    targets_ = std::move(shuffled_targets);
    this->current_index_ = 0;
  }

  void reset() override { this->current_index_ = 0; }

  size_t size() const override { return features_.size(); }
  std::vector<size_t> get_data_shape() const override { return {num_features_, 1, 1}; }
  size_t num_features() const { return num_features_; }
  size_t num_outputs() const { return num_outputs_; }
  bool is_regression() const { return is_regression_; }
  bool is_normalized() const { return is_normalized_; }

  std::vector<float> denormalize_targets(const std::vector<float> &normalized_targets) const {
    if (!is_normalized_ || !is_regression_) {
      return normalized_targets;
    }

    std::vector<float> denormalized(normalized_targets.size());
    for (size_t i = 0; i < normalized_targets.size() && i < target_means_.size(); ++i) {
      denormalized[i] = normalized_targets[i] * target_stds_[i] + target_means_[i];
    }
    return denormalized;
  }
};

}  // namespace tnn