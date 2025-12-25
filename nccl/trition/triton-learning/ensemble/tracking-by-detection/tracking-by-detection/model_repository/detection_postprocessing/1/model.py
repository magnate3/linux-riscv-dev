import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        print("Detection Postprocessing: initialization...")
        model_config = json.loads(args["model_config"])

        processed_detections = pb_utils.get_output_config_by_name(
            model_config, "processed_detections"
        )

        # Convert Triton types to numpy types
        self.processed_detections_dtype = pb_utils.triton_string_to_numpy(
            processed_detections["data_type"]
        )

    def execute(self, requests):

        def xywh2xyxy(x):
            x1 = x[0] - x[2] / 2
            y1 = x[1] - x[3] / 2
            x2 = x[0] + x[2] / 2
            y2 = x[1] + x[3] / 2
            return [x1, y1, x2, y2]

        MODEL_IMAGE_SIZE = (640, 640)

        def postprocess(outputs, image_shape, conf_thresold = 0.3, iou_threshold = 0.4):
            predictions = np.squeeze(outputs).T
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > conf_thresold, :]
            scores = scores[scores > conf_thresold]
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            boxes = predictions[:, :4]
            boxes = np.array(boxes)            
            boxes[:, 0::2] *= image_shape[1] / MODEL_IMAGE_SIZE[1]
            boxes[:, 1::2] *= image_shape[0] / MODEL_IMAGE_SIZE[0]

            if len(boxes) == 0:
                return np.empty([0, 6])
            
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresold, iou_threshold)
            detections = [np.append(
                xywh2xyxy(boxes[i]), [scores[i], class_ids[i]]) for i in indices] 
            return np.array(detections)

        responses = []
        for request in requests:
            detections = pb_utils.get_input_tensor_by_name(
                request, "detections"
            )
            postprocessing_params = pb_utils.get_input_tensor_by_name(
                request, "postprocessing_params"
            )

            detections = np.squeeze(detections.as_numpy(), axis=0)
            postprocessing_params = np.squeeze(postprocessing_params.as_numpy(), axis=0)
            processed_detections = postprocess(detections, postprocessing_params)
            processed_detections = np.expand_dims(processed_detections, axis=0)
            processed_detections = pb_utils.Tensor(
                "processed_detections", processed_detections.astype(self.processed_detections_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[processed_detections]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Detection Postprocessing: cleaning up...")
