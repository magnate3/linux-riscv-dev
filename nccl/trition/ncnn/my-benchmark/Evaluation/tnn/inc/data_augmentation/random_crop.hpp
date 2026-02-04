#pragma once

#include <random>

#include "augmentation.hpp"

namespace tnn {

/**
 * Random crop augmentation with padding
 */
class RandomCropAugmentation : public Augmentation {
public:
  RandomCropAugmentation(float probability = 0.5f, int padding = 4)
      : probability_(probability), padding_(padding) {
    this->name_ = "RandomCrop";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_ON_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<RandomCropAugmentation>(probability_, padding_);
  }

private:
  float probability_;
  int padding_;

  template <typename T>
  void apply_impl(Tensor &data, Tensor &labels) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    const auto shape = data->shape();
    if (shape.size() != 4) return;

    const size_t batch_size = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];

    std::uniform_int_distribution<int> crop_dist(0, 2 * padding_);

    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        int start_x = crop_dist(this->rng_);
        int start_y = crop_dist(this->rng_);
        apply_crop<T>(data, b, channels, height, width, start_x, start_y);
      }
    }
  }

  template <typename T>
  void apply_crop(Tensor &data, size_t batch_idx, size_t channels, size_t height, size_t width,
                  int start_x, int start_y) {
    const size_t padded_size = width + 2 * padding_;
    Tensor padded = Tensor::create<T>(std::vector<size_t>{1, channels, padded_size, padded_size});

    padded->fill(0.0);

    // Copy original image to center of padded image
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          padded->at<T>({0, c, h + padding_, w + padding_}) = data->at<T>({batch_idx, c, h, w});
        }
      }
    }

    // Crop from padded image
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          data->at<T>({batch_idx, c, h, w}) = padded->at<T>({0, c, start_y + h, start_x + w});
        }
      }
    }
  }
};

}  // namespace tnn
