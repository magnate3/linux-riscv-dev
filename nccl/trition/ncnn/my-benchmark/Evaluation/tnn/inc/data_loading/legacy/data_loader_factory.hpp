#pragma once

#include "data_loading/data_loader.hpp"

namespace tnn {
namespace legacy {

/**
 * A pair of data loaders for training and validation/testing
 */
struct DataLoaderPair {
  std::unique_ptr<BaseDataLoader> train;
  std::unique_ptr<BaseDataLoader> val;
};

/**
 * Factory class for creating data loaders by string name
 */
class DataLoaderFactory {
public:
  /**
   * Create a pair of data loaders (train and val) for a given dataset type
   * @param dataset_type Type of dataset (e.g., "mnist", "cifar10", "cifar100", "tiny_imagenet")
   * @param dataset_path Path to the dataset directory or file
   * @return DataLoaderPair containing the created loaders
   */
  static DataLoaderPair create(const std::string &dataset_type, const std::string &dataset_path);

  /**
   * Get list of available dataset types
   */
  static std::vector<std::string> available_loaders() {
    return {"mnist", "cifar10", "cifar100", "tiny_imagenet"};
  }
};
}  // namespace legacy
}  // namespace tnn