import argparse
import math

import cv2
import numpy as np
import tritonclient.http as httpclient

from utils import class_names, colors, nms, sigmoid, xywh2xyxy

parser = argparse.ArgumentParser(description="Process SegmentModel.")
parser.add_argument("-i", "--image", help="image paths to predict")


class Yolov8Seg:
    def __init__(self, host: str = "localhost", port: int = 8000, model_name: str = "yolov8_seg_onnx") -> None:
        self.url = f"{host}:{port}"
        self.model_name = model_name
        self.model_dtype = "FP32"
        self.inputs = []

        self.triton_client = httpclient.InferenceServerClient(
            url=self.url,
            verbose=False,
            network_timeout=600.0,  # this works when inference took too long (inference with cpu)
        )

        config = self.triton_client.get_model_config(
            model_name
        )  # if u use grpcclient add as_json = True and adapt ch, w, h
        self.input_channels = int(config["input"][0]["dims"][1])
        self.input_height = int(config["input"][0]["dims"][2])
        self.input_width = int(config["input"][0]["dims"][3])

        self.outputs = []
        self.outputs_name = [output["name"] for output in config["output"]]
        for output_name in self.outputs_name:
            self.outputs.append(httpclient.InferRequestedOutput(output_name))

    def prepare_inputs(self, img: np.array) -> None:
        """
        Prepares inputs for inference.

        :param img: input image as np.array.
        """
        input_tensor = self._preprocess(img)

        self.inputs.append(httpclient.InferInput(name="images", shape=input_tensor.shape, datatype="FP32"))
        self.inputs[0].set_data_from_numpy(input_tensor)

    def _preprocess(self, img: np.array) -> np.array:
        """
        Preprocess input image.

        :param img: input image as np.array.
        :return: preprocessed image as np.array.
        """
        self.img_height, self.img_width = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize
        input_tensor = np.array(img) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        return input_tensor

    def _model_inference(self, model_name: str) -> np.ndarray:
        """
        Performs inference model.

        :param model_name: model name.
        :return: inference result as np.array.
        """
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=self.inputs,
            outputs=self.outputs,
        )

        return results

    def process_box_output(self, box_output: np.ndarray) -> tuple:
        """
        Processes model output for object detection.

        :param box_output: model output for object detection.
        :return: containing BBox coordinates, Scores, ClassIDs, and MaskPredictions.
        """
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - 36

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:num_classes], axis=1)
        predictions = predictions[scores > self.conf_thresh, :]
        scores = scores[scores > self.conf_thresh]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_thresh)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions: np.ndarray, mask_output: np.ndarray) -> np.ndarray:
        """
        Processes model output for masks generation.

        :param mask_predictions: mask predictions.
        :param mask_output: mask prediction tensor dimension.
        :return: processed mask predictions.
        """
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes, (self.img_height, self.img_width), (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i, (scale_box, box) in enumerate(zip(scale_boxes, self.boxes)):
            scale_x1 = int(math.floor(scale_box[0]))
            scale_y1 = int(math.floor(scale_box[1]))
            scale_x2 = int(math.ceil(scale_box[2]))
            scale_y2 = int(math.ceil(scale_box[3]))

            x1 = int(math.floor(box[0]))
            y1 = int(math.floor(box[1]))
            x2 = int(math.ceil(box[2]))
            y2 = int(math.ceil(box[3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions: np.ndarray) -> np.ndarray:
        """
        Extracts boxes from predictions.

        :param box_predictions: predictions.
        :return: boxes representing in xyxy coordinates.
        """
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, (self.input_height, self.input_width), (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def main(self, img: np.ndarray, conf_thresh: float = 0.3, iou_thres: float = 0.7) -> None:
        self.img = img
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thres

        # Pre-process
        self.prepare_inputs(img)

        # Inference
        outputs = self._model_inference(self.model_name)
        outputs_data = [outputs.as_numpy(output_name) for output_name in self.outputs_name]

        # Post-process
        box_output, mask_output = outputs_data
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(box_output)
        self.mask_maps = self.process_mask_output(mask_pred, mask_output)

        return {"boxes": self.boxes, "scores": self.scores, "cls_ids": self.class_ids, "masks": self.mask_maps}

    @staticmethod
    def rescale_boxes(boxes: np.ndarray, input_shape: tuple[int, int], image_shape: tuple[int, int]) -> np.ndarray:
        """
        Rescales bounding boxes to original image dimensions

        :param boxes: bounding boxes.
        :param input_shape: input shape YOLO model.
        :param image_shape: original image shape.
        :return: rescaled bounding boxes.
        """
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

    def draw_detections(self, mask_alpha: float = 0.3) -> np.ndarray:
        """
        Draws object on the image.

        :param mask_alpha: opacity of the mask overlay. Default is 0.3.
        :return: image with object masks overlaid on it.
        """
        size = min([self.img_height, self.img_width]) * 0.0007
        text_thickness = int(min([self.img_height, self.img_width]) * 0.001)

        mask_img = self.draw_masks(mask_alpha, self.mask_maps)

        # Draw bounding boxes and labels of detections
        labels = []
        for box, score, class_id in zip(self.boxes, self.scores, self.class_ids):
            color = colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

            label = class_names[class_id]
            labels.append(label)
            caption = f"{label} {int(score * 100)}%"
            (tw, th), _ = cv2.getTextSize(
                text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=text_thickness
            )
            th = int(th * 1.2)

            cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

            cv2.putText(
                mask_img,
                caption,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        return mask_img

    def draw_masks(self, mask_alpha: float = 0.3, mask_maps=None) -> np.ndarray:
        """
        Draw masks on the image.

        :param mask_alpha: opacity of the mask overlay. Default is 0.3.
        :param mask_maps: mask predictions. Default is None.
        :return: image with object masks overlaid on it.
        """
        mask_img = self.img.copy()

        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(self.boxes, self.class_ids)):
            color = colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw fill mask image
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

        return cv2.addWeighted(mask_img, mask_alpha, self.img, 1 - mask_alpha, 0)

    def visualize_contours(self, vis: bool = False, save: bool = False) -> None:
        """
        Visualize contours of detected objects on image.

        :param vis: If True display image with contours.
        :param save: If True save image with contours to file.
        """
        for ind, cls_id in enumerate(self.class_ids):
            contours, _ = cv2.findContours(
                (self.mask_maps[ind] * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(self.img, contours, -1, colors[cls_id], 3)

            label = f"{class_names[cls_id]}: {self.scores[ind]:.2f}"
            text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)

            cv2.rectangle(
                self.img,
                (int(contours[0][0][0][0]) - text_size[0] // 2 - 10, int(contours[0][0][0][1]) - text_size[1] - 10),
                (int(contours[0][0][0][0]) + text_size[0] // 2 + 5, int(contours[0][0][0][1] + 5)),
                colors[cls_id],
                -1,
            )
            cv2.putText(
                self.img,
                label,
                (int(contours[0][0][0][0]) - text_size[0] // 2, int(contours[0][0][0][1]) - 5),
                0,
                0.7,
                (255, 255, 255),
                2,
            )

        # Img show
        if vis:
            cv2.imshow("demo_co", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            cv2.imwrite("./data/demo_co.jpg", self.img)


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = "yolov8_seg_onnx"

    img = cv2.imread(args.image)

    model = Yolov8Seg(model_name=model_name)

    results = model.main(img)

    cv2.imwrite("./data/demo_dd.jpg", model.draw_detections())
    model.visualize_contours(save=True)
