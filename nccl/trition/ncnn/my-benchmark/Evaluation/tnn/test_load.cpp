#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/example_models.hpp"
#include "nn/layers.hpp"
#include "nn/train.hpp"
#include "type/type.hpp"
#include "utils/env.hpp"

using namespace std;
using namespace tnn;

signed main() {
  string dataset_name = Env::get<std::string>("DATASET_NAME", "");
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  int batch_size = 1;
  string dataset_path = Env::get<std::string>("DATASET_PATH", "data");
  auto [train_loader, val_loader] =
      DataLoaderFactory::create(dataset_name, dataset_path, DType_t::FP16);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << model_name << endl;
    return 1;
  }

  cout << "Training model on device: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

  float lr_initial = Env::get("LR_INITIAL", 0.001f);
  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer = OptimizerFactory::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f);
  auto scheduler = SchedulerFactory::create_step_lr(optimizer.get(), 10, 0.1f);

  try {
    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, train_config);
  } catch (const std::exception &e) {
    cerr << "Training failed: " << e.what() << endl;
    return 1;
  }
  cout << "Training batches: " << train_loader.size() << endl;
  cout << "Validation batches: " << val_loader.size() << endl;

  vector<size_t> data_shape = train_loader->get_data_shape();
  data_shape.insert(data_shape.begin(), batch_size);  // add batch dimension
  return 0;
}
