import typing as T
import json

import cv2
import numpy as np

import triton_python_backend_utils as pb_utils


def postprocess(
    output: np.ndarray,
    new_shape: T.Tuple[int, int],
    ori_shape: T.Tuple[int, int],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Postprocess the output of the network.

    Args:
        output (np.ndarray): The output of the network (1, 84, 8400).
        new_shape (T.Tuple[int, int]): The new shape of the image (height, width).
        ori_shape (T.Tuple[int, int]): The original shape of the image (height, width).
        conf_thres (float, optional): The confidence threshold. Defaults to 0.25.
        iou_thres (float, optional): The IoU threshold. Defaults to 0.45.

    Returns:
        T.Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores, and class IDs.
    """
    assert output.shape == (1, 84, 8400)
    output = output[0]  # (84, 8400)
    output = output.transpose(1, 0)  # (8400, 84)

    bboxes = output[:, :4]  # (8400, 4) in cxcywh
    # cxcywh to xywh
    bboxes[..., 0] -= bboxes[..., 2] / 2
    bboxes[..., 1] -= bboxes[..., 3] / 2
    scores = np.max(output[:, 4:], axis=1)  # (8400,)
    class_ids = np.argmax(output[:, 4:], axis=1)  # (8400,)

    # Batched NMS
    keep = cv2.dnn.NMSBoxesBatched(
        bboxes=bboxes,  # type: ignore
        scores=scores,
        class_ids=class_ids,
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
        top_k=300,
    )
    bboxes = bboxes[keep]  # type: ignore
    scores = scores[keep]
    class_ids = class_ids[keep]
    # xywh to xyxy
    bboxes[..., 2] += bboxes[..., 0]
    bboxes[..., 3] += bboxes[..., 1]

    # Scale and clip bboxes.
    bboxes = scale_boxes(bboxes, new_shape, ori_shape)
    return bboxes, scores, class_ids


def scale_boxes(bboxes: np.ndarray, new_shape: T.Tuple[int, int], ori_shape: T.Tuple[int, int]) -> np.ndarray:
    """Rescale bounding boxes to the original shape.

    Preprocess: ori_shape => new_shape
    Postprocess: new_shape => ori_shape

    Args:
        bboxes (np.ndarray): The bounding boxes in (x1, y1, x2, y2) format.
        new_shape (T.Tuple[int, int]): The new shape of the image (height, width).
        ori_shape (T.Tuple[int, int]): The original shape of the image (height, width).

    Returns:
        np.ndarray: The rescaled and clipped bounding boxes.
    """
    # calculate from ori_shape
    gain = min(new_shape[0] / ori_shape[0], new_shape[1] / ori_shape[1])  # gain  = old / new
    pad = (
        round((new_shape[1] - ori_shape[1] * gain) / 2 - 0.1),
        round((new_shape[0] - ori_shape[0] * gain) / 2 - 0.1),
    )  # wh padding

    bboxes[..., [0, 2]] -= pad[0]  # x padding
    bboxes[..., [1, 3]] -= pad[1]  # y padding

    bboxes /= gain
    # Clip bounding box coordinates to the image shape.
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clip(0, ori_shape[1])  # x1, x2
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clip(0, ori_shape[0])  # y1, y2
    return bboxes


class TritonPythonModel:
    def initialize(self, args: T.Dict[str, T.Any]) -> None:
        model_config = json.loads(args["model_config"])

        bboxes_config = pb_utils.get_output_config_by_name(model_config, "detection_postprocessing_bboxes")
        scores_config = pb_utils.get_output_config_by_name(model_config, "detection_postprocessing_scores")
        ids_config = pb_utils.get_output_config_by_name(model_config, "detection_postprocessing_ids")

        self.bboxes_dtype = pb_utils.triton_string_to_numpy(bboxes_config["data_type"])
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])
        self.ids_dtype = pb_utils.triton_string_to_numpy(ids_config["data_type"])

    def execute(self, requests):
        responses = []

        for request in requests:
            detections = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_detections"
            ).as_numpy()

            orig_shape = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_shape"
            ).as_numpy()
            new_shape = (640, 640)

            bboxes, scores, class_ids = postprocess(detections, new_shape, orig_shape)

            bboxes_tensor = pb_utils.Tensor(
                "detection_postprocessing_bboxes", bboxes.astype(self.bboxes_dtype)
            )
            scores_tensor = pb_utils.Tensor(
                "detection_postprocessing_scores", scores.astype(self.scores_dtype)
            )
            ids_tensor = pb_utils.Tensor("detection_postprocessing_ids", class_ids.astype(self.ids_dtype))

            response = pb_utils.InferenceResponse(output_tensors=[bboxes_tensor, scores_tensor, ids_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        print("Cleaning up...")
