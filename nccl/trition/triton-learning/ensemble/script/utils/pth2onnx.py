import torch
import argparse
from model import STRModel

# 创建一个解析器对象
parser = argparse.ArgumentParser(description='pytorch to onnx')

# 添加命令行参数
parser.add_argument('--input', type=str, help='input file, pytorch model file')
parser.add_argument('--output', type=str, help='output file, onnx model file')
parser.add_argument('--batch', action='store_true')

# 解析命令行参数
args = parser.parse_args()


# Create PyTorch Model Object
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# Load model weights from external file
state = torch.load(args.input)
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

if args.batch:
    # Create ONNX file by tracing model
    trace_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(
        model,
        trace_input,
        args.output,
        verbose=True,
        dynamic_axes={"input.1": [0], "308": [0]},
    )
else:
    # Create ONNX file by tracing model
    trace_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(model, trace_input, args.output, verbose=True)

