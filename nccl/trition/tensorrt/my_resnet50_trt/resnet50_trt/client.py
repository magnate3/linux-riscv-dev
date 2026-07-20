# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os, sys
import numpy as np
import json
import tritongrpcclient
from torchvision import transforms
import argparse
import time
import tritonclient.http as httpclient
from PIL import Image


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    This is a typical approach you'd like to use in DALI backend.
    DALI performs image decoding, therefore this way the processing
    can be fully offloaded to the GPU.
    """
    return np.fromfile(img_path, dtype='float32')


# preprocessing function
#def rn50_preprocess(img_path="img1.jpg"):
def rn50_preprocess(img_path="mug.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return np.expand_dims(preprocess(img).numpy(), axis=0)
with open("./model_repository/resnet50_trt/labels.txt") as f:
     labels_dict = {idx: line.strip() for idx, line in enumerate(f)}
output_name = "output"
transformed_img = rn50_preprocess()

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("input", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("output", binary_data=True)

start_time = time.time()
# Querying the server
results = client.infer(model_name="resnet50_trt", inputs=[inputs], outputs=[outputs])
latency = time.time() - start_time
output0_data = results.as_numpy(output_name)
#print(outputs)
#print(output0_data)
maxs = np.argmax(output0_data, axis=1)


print(f"{latency * 1000}ms class: {labels_dict[maxs[0]]}")
