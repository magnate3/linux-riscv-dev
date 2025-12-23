import onnx
# Load the ONNX model
#model = onnx.load("resnet101-v1-7/resnet101-v1-7.onnx")
#model = onnx.load("resnet101-v1-7.onnx")
model = onnx.load("model_repository/resnet18/1/model.onnx")
#model = onnx.load("resnet18_fp32.onnx")
# Check the model (raises an error if invalid)
onnx.checker.check_model(model)
