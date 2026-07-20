import typing as T
import json

import numpy as np
import cv2

import triton_python_backend_utils as pb_utils


def preprocess(image: np.ndarray, new_shape: T.Tuple[int, int]) -> np.ndarray:
    """Preprocess the input image.

    Args:
        image (np.ndarray): The input image (h, w, 3).
        new_shape (T.Tuple[int, int]): The new shape of the image (height, width).

    Returns:
        np.ndarray: The preprocessed image (1, 3, height, width).
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize but keep aspect ratio
    h, w = image.shape[:2]
    height, width = new_shape
    ratio = min(height / h, width / w)
    image = cv2.resize(image, (int(w * ratio), int(h * ratio)))
    # Pad to new_shape
    dh = (height - image.shape[0]) / 2
    dw = (width - image.shape[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, ...].astype(np.float32) / 255.0
    return image


class TritonPythonModel:
    def initialize(self, args: T.Dict[str, T.Any]) -> None:
        model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(model_config, "detection_preprocessing_output")
        shape_config = pb_utils.get_output_config_by_name(model_config, "detection_preprocessing_shape")

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        self.shape_dtype = pb_utils.triton_string_to_numpy(shape_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "detection_preprocessing_input").as_numpy()
            shape = np.array([image.shape[0], image.shape[1]])  # (height, width)
            image = preprocess(image, (640, 640))

            output_tensor = pb_utils.Tensor("detection_preprocessing_output", image.astype(self.output_dtype))
            shape_tensor = pb_utils.Tensor("detection_preprocessing_shape", shape.astype(self.shape_dtype))

            response = pb_utils.InferenceResponse(output_tensors=[output_tensor, shape_tensor])
            responses.append(response)
        return responses

    def finalize(self) -> None:
        print("Cleaning up...")
