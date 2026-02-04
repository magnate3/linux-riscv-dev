#pragma once

#include <algorithm>
#include <memory>
#include <random>

#include "tensor/tensor.hpp"

namespace tnn {

class Augmentation;
class AugmentationStrategy;

/**
 * Abstract base class for all augmentation operations
 */
class Augmentation {
public:
  Augmentation() = default;
  virtual ~Augmentation() = default;

  virtual void apply(Tensor &data, Tensor &labels) = 0;
  virtual std::unique_ptr<Augmentation> clone() const = 0;

  void set_name(const std::string &name) { name_ = name; }
  std::string get_name() const { return name_; }

protected:
  std::string name_;
  mutable std::mt19937 rng_{std::random_device{}()};
};

}  // namespace tnn

// Include concrete augmentation implementations
#include "brightness.hpp"
#include "contrast.hpp"
#include "cutout.hpp"
#include "gaussian_noise.hpp"
#include "horizontal_flip.hpp"
#include "normalization.hpp"
#include "random_crop.hpp"
#include "rotation.hpp"
#include "vertical_flip.hpp"

namespace tnn {

class AugmentationStrategy {
public:
  AugmentationStrategy() = default;

  AugmentationStrategy(const AugmentationStrategy &other) {
    for (const auto &aug : other.augmentations_) {
      augmentations_.emplace_back(aug->clone());
    }
  }

  /**
   * Apply augmentations in the order they were added
   */
  void apply(Tensor &data, Tensor &labels) {
    for (auto &aug : augmentations_) {
      aug->apply(data, labels);
    }
  }

  void add_augmentation(const Augmentation &augmentation) {
    augmentations_.emplace_back(augmentation.clone());
  }

  void add_augmentation(std::unique_ptr<Augmentation> augmentation) {
    augmentations_.emplace_back(std::move(augmentation));
  }

  void remove_augmentation(size_t index) {
    if (index >= augmentations_.size()) return;
    augmentations_.erase(augmentations_.begin() + index);
  }

  void remove_augmentation(const std::string &name) {
    augmentations_.erase(std::remove_if(augmentations_.begin(), augmentations_.end(),
                                        [&name](const std::unique_ptr<Augmentation> &aug) {
                                          return aug->get_name() == name;
                                        }),
                         augmentations_.end());
  }

  void set_augmentations(const std::vector<std::unique_ptr<Augmentation>> &augs) {
    augmentations_.clear();
    for (const auto &aug : augs) {
      augmentations_.emplace_back(aug->clone());
    }
  }

  void clear_augmentations() { augmentations_.clear(); }

  size_t size() const { return augmentations_.size(); }

  const std::vector<std::unique_ptr<Augmentation>> &get_augmentations() const {
    return augmentations_;
  }

protected:
  std::vector<std::unique_ptr<Augmentation>> augmentations_;
};

class AugmentationBuilder {
public:
  AugmentationBuilder() = default;

  AugmentationBuilder &horizontal_flip(float probability = 0.5f) {
    strategy_.add_augmentation(std::make_unique<HorizontalFlipAugmentation>(probability));
    return *this;
  }

  AugmentationBuilder &vertical_flip(float probability = 0.5f) {
    strategy_.add_augmentation(std::make_unique<VerticalFlipAugmentation>(probability));
    return *this;
  }

  AugmentationBuilder &rotation(float probability = 0.5f, float max_angle_degrees = 15.0f) {
    strategy_.add_augmentation(
        std::make_unique<RotationAugmentation>(probability, max_angle_degrees));
    return *this;
  }

  AugmentationBuilder &brightness(float probability = 0.5f, float range = 0.2f) {
    strategy_.add_augmentation(std::make_unique<BrightnessAugmentation>(probability, range));
    return *this;
  }

  AugmentationBuilder &contrast(float probability = 0.5f, float range = 0.2f) {
    strategy_.add_augmentation(std::make_unique<ContrastAugmentation>(probability, range));
    return *this;
  }

  AugmentationBuilder &gaussian_noise(float probability = 0.3f, float std_dev = 0.05f) {
    strategy_.add_augmentation(std::make_unique<GaussianNoiseAugmentation>(probability, std_dev));
    return *this;
  }

  AugmentationBuilder &random_crop(float probability = 0.5f, int padding = 4) {
    strategy_.add_augmentation(std::make_unique<RandomCropAugmentation>(probability, padding));
    return *this;
  }

  AugmentationBuilder &cutout(float probability = 0.5f, int cutout_size = 8) {
    strategy_.add_augmentation(std::make_unique<CutoutAugmentation>(probability, cutout_size));
    return *this;
  }

  AugmentationBuilder &normalize(const std::array<float, 3> &mean = {0.485f, 0.456f, 0.406f},
                                 const std::array<float, 3> &std = {0.229f, 0.224f, 0.225f}) {
    strategy_.add_augmentation(std::make_unique<NormalizationAugmentation>(mean, std));
    return *this;
  }

  AugmentationBuilder &custom_augmentation(std::unique_ptr<Augmentation> augmentation) {
    strategy_.add_augmentation(std::move(augmentation));
    return *this;
  }

  std::unique_ptr<AugmentationStrategy> build() {
    return std::make_unique<AugmentationStrategy>(strategy_);
  }

private:
  AugmentationStrategy strategy_;
};

}  // namespace tnn