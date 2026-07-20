import os
import sys

try: # for 'model' command
    import torch.nn as nn

    class TestNet(nn.Module):
        def __init__(self, in_size=3 * 224*224, hidden_size=64, out_size=3):
            super(TestNet, self).__init__()
            self.linear1 = nn.Linear(in_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, out_size)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = x.view(x.size(0), -1) # [batch, 3, 224, 224] => [batch, 3 * 224 * 224]
            x = self.linear1(x) # [batch, 3 * 244 * 244] => [batch, 64]
            x = self.linear2(x) # [batch, 64] => [batch, 3]
            x = self.softmax(x) # [batch, 3] => [batch, 3]
            return x
except ImportError:
    pass
from tensorrtserver.api import InferContext, ServerHealthContext, ProtocolType
try: # for 'infer' command
    import numpy as np
    from tensorrtserver.api import InferContext, ServerHealthContext, ProtocolType
except ImportError:
    pass

def create_model():
    import torch
    import torch.nn as nn
    import torch.onnx

    model = TestNet()

    batch = torch.rand((5, 3, 224, 224))

    # Export the model
    torch.onnx.export(model,                   # model being run
                    batch,                     # model input (or a tuple for multiple inputs)
                    "models/test_model/1/model.onnx",               # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


def init_trtis(url, protocol, model_name, verbose=True):
    ctx = InferContext(url, protocol, model_name, 1)
    return ctx

def check_model_health(triton_uri, protocol):
    health_context = ServerHealthContext(triton_uri, protocol)

    max_retry = 2 * 60 * 5  # about 5 minutes
    trial = 0
    while True:
        try:
            trial += 1
            print("Trying to check TRITON server health ...")
            if health_context.is_ready():
                break
            raise
        except Exception as ex:
            if trial >= max_retry:
                print('Failed to get server status'.format(ex))
                raise
            else:
                time.sleep(0.5)

def infer_batch(ctx, batch, classes=3):
    batch_size = len(batch)
    results = ctx.run({'input': batch},
                      {'output': (InferContext.ResultFormat.CLASS, classes)}, batch_size)
    return results

def infer():
    triton_uri = os.environ['NVIDIA_TRITONURI']
    protocol = ProtocolType.HTTP
    check_model_health(triton_uri, protocol)
    ctx = init_trtis(triton_uri, protocol, 'test_model')

    batch_size = 10

    batch = []
    for i in range(batch_size):
        batch.append(np.random.randn(3, 224, 224).astype(np.float32))

    predictions = infer_batch(ctx, batch)
    for pred in predictions['output']:
        print("index: {} value: {} class: {}".format(*pred))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify command!")
        sys.exit(1)
    if sys.argv[1] == 'model':
        create_model()
    elif sys.argv[1] == 'infer':
        infer()
    else:
        print("Please specify either 'model' or 'inference'!")





