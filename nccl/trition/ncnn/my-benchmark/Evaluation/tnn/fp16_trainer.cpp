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
  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  std::string model_name = Env::get<std::string>("MODEL_NAME", "cifar10_resnet9");
  std::string model_path = Env::get<std::string>("MODEL_PATH", "");

  std::string device_str = Env::get<std::string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);

  std::unique_ptr<Sequential> model;
  if (!model_path.empty()) {
    cout << "Loading model from: " << model_path << endl;
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open model file");
    }
    model = load_state<Sequential>(file, device);
    file.close();
  } else {
    cout << "Creating model: " << model_name << endl;
    try {
      Sequential temp_model = ExampleModels::create(model_name);
      model = std::make_unique<Sequential>(std::move(temp_model));
    } catch (const std::exception &e) {
      cerr << "Error creating model: " << e.what() << endl;
      cout << "Available models are: ";
      for (const auto &name : ExampleModels::available_models()) {
        cout << name << "\n";
      }
      cout << endl;
      return 1;
    }
    model->set_device(device);
    model->set_param_dtype(DType_t::FP16);
    model->set_io_dtype(DType_t::FP16);
    model->init();
  }

  string dataset_name = Env::get<std::string>("DATASET_NAME", "");
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
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

  return 0;
}