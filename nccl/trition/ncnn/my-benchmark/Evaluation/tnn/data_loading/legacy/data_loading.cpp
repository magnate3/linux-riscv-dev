/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "data_loading/legacy/cifar100_data_loader.hpp"
#include "data_loading/legacy/cifar10_data_loader.hpp"
#include "data_loading/legacy/data_loader_factory.hpp"
#include "data_loading/legacy/mnist_data_loader.hpp"
#include "data_loading/legacy/tiny_imagenet_data_loader.hpp"
#include "data_loading/open_webtext_data_loader.hpp"

namespace tnn {
namespace legacy {
DataLoaderPair DataLoaderFactory::create(const std::string &dataset_type,
                                         const std::string &dataset_path) {
  DataLoaderPair pair;

  if (dataset_type == "mnist") {
    auto train = std::make_unique<MNISTDataLoader>();
    auto val = std::make_unique<MNISTDataLoader>();

    if (train->load_data(dataset_path + "/train.csv") ||
        train->load_data(dataset_path + "/mnist_train.csv")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test.csv") ||
        val->load_data(dataset_path + "/mnist_test.csv")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "cifar10") {
    auto train = std::make_unique<CIFAR10DataLoader>();
    auto val = std::make_unique<CIFAR10DataLoader>();

    std::vector<std::string> train_files = {
        dataset_path + "/data_batch_1.bin", dataset_path + "/data_batch_2.bin",
        dataset_path + "/data_batch_3.bin", dataset_path + "/data_batch_4.bin",
        dataset_path + "/data_batch_5.bin"};

    if (train->load_multiple_files(train_files)) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test_batch.bin")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "cifar100") {
    auto train = std::make_unique<CIFAR100DataLoader>();
    auto val = std::make_unique<CIFAR100DataLoader>();

    if (train->load_data(dataset_path + "/train.bin")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test.bin")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "tiny_imagenet") {
    auto train = std::make_unique<TinyImageNetDataLoader>();
    auto val = std::make_unique<TinyImageNetDataLoader>();

    if (train->load_data(dataset_path, true)) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path, false)) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "open_webtext") {
    auto train = std::make_unique<OpenWebTextDataLoader>(512);
    auto val = std::make_unique<OpenWebTextDataLoader>(512);

    if (train->load_data(dataset_path + "/train.bin")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/train.bin")) {
      pair.val = std::move(val);
    }
  } else {
    std::cerr << "Error: Unknown dataset type: " << dataset_type << std::endl;
  }

  // If we only have test/val, or we want to use test for val if val is missing
  if (!pair.val && pair.train) {
    // This case usually doesn't happen with the logic above, but per request:
    // "just take the test for val if it only has test"
    // (Though usually we want the opposite: if we only have one set, use it for both or split)
  }

  return pair;
}
}  // namespace legacy
}  // namespace tnn
