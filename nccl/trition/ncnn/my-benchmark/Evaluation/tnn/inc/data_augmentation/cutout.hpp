#pragma once

#include <random>

#include "augmentation.hpp"

namespace tnn {

/**
 * Cutout augmentation (random erasing)
 */
class CutoutAugmentation : public Augmentation {
public:
  CutoutAugmentation(float probability = 0.5f, int cutout_size = 8)
      : probability_(probability), cutout_size_(cutout_size) {
    this->name_ = "Cutout";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_ON_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<CutoutAugmentation>(probability_, cutout_size_);
  }

private:
  float probability_;
  int cutout_size_;

  template <typename T>
  void apply_impl(Tensor &data, Tensor &labels) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    const auto shape = data->shape();
    if (shape.size() != 4) return;

    const size_t batch_size = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        std::uniform_int_distribution<size_t> x_dist(0, width - cutout_size_);
        std::uniform_int_distribution<size_t> y_dist(0, height - cutout_size_);

        size_t x = x_dist(this->rng_);
        size_t y = y_dist(this->rng_);

        for (size_t c = 0; c < channels; ++c) {
          for (size_t h = y; h < y + cutout_size_ && h < height; ++h) {
            for (size_t w = x; w < x + cutout_size_ && w < width; ++w) {
              data->at<T>({b, c, h, w}) = static_cast<float>(0);
            }
          }
        }
      }
    }
  }
};

}  // namespace tnn
