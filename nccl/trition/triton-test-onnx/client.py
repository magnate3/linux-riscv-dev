# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype



# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")


outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=10)

# Querying the server
#results = client.infer(model_name="resnet18", inputs=[inputs], outputs=[outputs])
#results = client.infer(model_name="resnet101-v1-7", inputs=[inputs], outputs=[outputs])
#results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
#inference_output = results.as_numpy("fc6_1").astype(str)
batch_size = 10

batch = []
inputs = httpclient.InferInput("data_0", np.random.randn(3, 224, 224).astype(np.float32), datatype="FP32")
#inputs.set_data_from_numpy(transformed_img, binary_data=True)
predictions = client.infer(model_name="test_model", inputs=inputs, outputs=[outputs])
for pred in predictions['output']:
    print("index: {} value: {} class: {}".format(*pred))

print(np.squeeze(inference_output)[:5])
