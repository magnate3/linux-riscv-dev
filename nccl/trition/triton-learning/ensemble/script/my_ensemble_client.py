# import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt
from PIL import Image
import time
import requests
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2, service_pb2_grpc
class TritonClient:
    def __init__(self, url="localhost:8001"):
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=self.url)

    def inference(self, model_name=None, input_name=None, input_dtype=None, path=None):
        assert model_name is not None, "Insert the model for inference"
        assert input_name is not None, "Insert input name for inference"
        assert input_dtype is not None, "Insert data type for inference"
        assert path is not None, "Insert one image for inference"

        image_data = np.fromfile(path, dtype="uint8")
        image_data = np.expand_dims(image_data, axis=0)

        inputs = [grpcclient.InferInput(input_name, image_data.shape, input_dtype)]
        inputs[0].set_data_from_numpy(image_data)

        results = self.client.infer(model_name=model_name, inputs=inputs)
        output_data = results.as_numpy("recognized_text").astype(str)
        print(output_data)
if __name__ == "__main__":
    triton_client = TritonClient()
    triton_client.inference(
        model_name="ensemble_model",
        input_name="input_image",
        input_dtype="UINT8",
        path="./sample.jpg"
    )
