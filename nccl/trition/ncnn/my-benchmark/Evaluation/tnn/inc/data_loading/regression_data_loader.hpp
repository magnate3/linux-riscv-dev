/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "data_loader.hpp"

namespace tnn {
/**
 * Specialized base class for regression datasets
 * Provides common functionality for continuous target prediction
 */
class RegressionDataLoader : public BaseDataLoader {
public:
  virtual ~RegressionDataLoader() = default;

  /**
   * Get number of input features
   */
  virtual size_t get_num_features() const = 0;

  /**
   * Get number of output targets
   */
  virtual size_t get_num_outputs() const = 0;

  /**
   * Check if data is normalized
   */
  virtual bool is_normalized() const = 0;

  /**
   * Get feature normalization statistics (optional)
   */
  virtual std::vector<float> get_feature_means() const { return {}; }
  virtual std::vector<float> get_feature_stds() const { return {}; }

  /**
   * Get target normalization statistics (optional)
   */
  virtual std::vector<float> get_target_means() const { return {}; }
  virtual std::vector<float> get_target_stds() const { return {}; }
};

}  // namespace tnn