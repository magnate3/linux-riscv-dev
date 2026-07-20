import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        print('Detection Preprocessing: initialization...')
        model_config = json.loads(args["model_config"])

        preprocessed_image = pb_utils.get_output_config_by_name(
            model_config, 'preprocessed_image'
        )

        preprocessing_params = pb_utils.get_output_config_by_name(
            model_config, 'preprocessing_params'
        )

        self.preprocessed_image_dtype = pb_utils.triton_string_to_numpy(
            preprocessed_image['data_type']
        )
        self.preprocessing_params_dtype = pb_utils.triton_string_to_numpy(
            preprocessing_params['data_type']
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            image = pb_utils.get_input_tensor_by_name(
                request, 'image'
            )

            image = image.as_numpy()
            image = np.squeeze(image, axis=0)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (640, 640))
            image_normalized = image_resized / 255.0
            image_preprocessed = image_normalized.transpose(2, 0, 1)

            image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
            preprocessing_params = np.expand_dims(
                np.array(image.shape), axis=0)

            image_preprocessed = pb_utils.Tensor(
                'preprocessed_image', image_preprocessed.astype(
                    self.preprocessed_image_dtype)
            )
            preprocessing_params = pb_utils.Tensor(
                'preprocessing_params', preprocessing_params.astype(
                    self.preprocessing_params_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[image_preprocessed, preprocessing_params]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Detection Preprocessing: cleaning up...')
