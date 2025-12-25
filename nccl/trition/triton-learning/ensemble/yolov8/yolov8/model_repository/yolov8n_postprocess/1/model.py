# Copyright (C) 2025 YIQISOFT
#
# SPDX-License-Identifier: Apache-2.0
#

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Python Backend for YOLO model post-processing.
    This script receives the raw output from the ONNX model, performs Non-Max Suppression (NMS),
    and outputs the formatted bounding box information.
    """

    def initialize(self, args):
        """
        Called once when the model is loaded.
        Used for loading configuration or one-time setup.
        """
        # Parse the model configuration file (config.pbtxt)
        self.model_config = json.loads(args['model_config'])
        
        # Get the output tensor configuration to determine the output data type
        output_config = pb_utils.get_output_config_by_name(self.model_config, "bboxes")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])
        
        print("YOLO postprocess model initialized.")

    def _iou(self, box1, boxes):
        """
        Calculates the IoU (Intersection over Union) between a single bounding box and a set of bounding boxes.
        
        Parameters:
        box1: shape (4,) - [xmin, ymin, xmax, ymax]
        boxes: shape (N, 4) - [[xmin, ymin, xmax, ymax], ...]
        
        Returns:
        IoUs: shape (N,)
        """
        # Calculate coordinates of the intersection area
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        # Calculate the intersection area
        intersection_area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        
        # Calculate the area of each box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Calculate the union area
        union_area = box1_area + boxes_area - intersection_area
        
        # Calculate IoU, avoiding division by zero
        return intersection_area / (union_area + 1e-7)

    def _non_max_suppression(self, predictions, conf_thres=0.5, iou_thres=0.5):
        """
        Performs Non-Max Suppression (NMS) on the raw model output.
        Adapted for YOLOv8 format: [cx, cy, w, h, class_prob_0, class_prob_1, ...]
        """
        # 1. Class scores start directly from column 4
        # 4 is the number of coordinates (4)
        class_scores = predictions[:, 4:]  
        
        # 2. Find the highest scoring class ID and its corresponding score for each box
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)
        
        # 3. Filter: only keep predictions where the max class score is above the threshold
        keep_mask = max_scores > conf_thres
        
        # Filter predictions (coordinates) and scores/ids
        predictions = predictions[keep_mask]
        class_ids = class_ids[keep_mask]
        max_scores = max_scores[keep_mask]

        if not predictions.shape[0]:
            return np.array([])
        
        # 4. Coordinate Transformation: (center_x, center_y, width, height) -> (xmin, ymin, xmax, ymax)
        box = predictions[:, :4]
        cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        
        # Combine into a new array for easy processing
        # Format: [xmin, ymin, xmax, ymax, score, class_id]
        boxes_with_scores = np.column_stack((xmin, ymin, xmax, ymax, max_scores, class_ids))

        # 6. Perform NMS per class
        final_boxes = []
        unique_class_ids = np.unique(boxes_with_scores[:, 5])
        
        for class_id in unique_class_ids:
            class_mask = (boxes_with_scores[:, 5] == class_id)
            class_boxes = boxes_with_scores[class_mask]
            
            # Sort by confidence score in descending order
            sorted_indices = np.argsort(class_boxes[:, 4])[::-1]
            class_boxes = class_boxes[sorted_indices]
            
            while class_boxes.shape[0] > 0:
                # Select the box with the highest confidence
                best_box = class_boxes[0]
                final_boxes.append(best_box)
                
                if class_boxes.shape[0] == 1:
                    break
                
                # Calculate IoU
                ious = self._iou(best_box[:4], class_boxes[1:, :4])
                
                # Keep boxes with IoU less than the threshold
                class_boxes = class_boxes[1:][ious < iou_thres]
        
        return np.array(final_boxes)

    def execute(self, requests):
        """
        In every inference request, this function is called.
        """
        responses = []
        
        # Iterate over each request in the batch
        for req_idx, request in enumerate(requests):
            # Get the input tensor named "raw_output" from the request
            raw_output = pb_utils.get_input_tensor_by_name(request, "raw_output").as_numpy()
            
            # ❗ Temporary Code for Debugging/Verification ❗
            print(f"Raw Output Shape: {raw_output.shape}")
            # Print max and min values in the raw output
            print(f"Raw Output Max: {np.max(raw_output):.4f}, Min: {np.min(raw_output):.4f}")
            # Print the slice where objectness scores would traditionally be (index 4)
            # Note: For YOLOv8, this column is the first class score, not a separate objectness score.
            # Traditional YOLOv5/v7: This is the 5th row (index 4)
            conf_slice = raw_output[0, 4, :] 
            print(f"Objectness Slice Max: {np.max(conf_slice):.4f}")
            # ❗ End Temporary Code ❗
            
            # Remove the batch dimension and transpose to match NMS input expectation
            # Transforms from e.g., (1, 84, 8400) to (8400, 84)
            predictions = np.squeeze(raw_output, axis=0).T
            
            # Perform NMS and other post-processing steps
            # Note: The thresholds (0.5, 0.5) are now set here and override initial config
            final_boxes = self._non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.5)

            # Format the output to match the client's expectation:
            # [image_id, label_id, conf, xmin, ymin, xmax, ymax]
            
            if final_boxes.shape[0] > 0:
                # final_boxes columns: [xmin, ymin, xmax, ymax, conf, class_id]
                image_id_col = np.full((final_boxes.shape[0], 1), req_idx)
                # Reorder columns: [class_id, conf, xmin, ymin, xmax, ymax]
                reordered_boxes = final_boxes[:, [5, 4, 0, 1, 2, 3]] 
                # Stack with image_id
                output_data = np.hstack((image_id_col, reordered_boxes))
            else:
                # If no objects are detected, return an empty array of shape (0, 7)
                output_data = np.empty((0, 7), dtype=self.output_dtype)

            # Create a Triton output tensor
            out_tensor = pb_utils.Tensor("bboxes", output_data.astype(self.output_dtype))
            
            # Create a response for the current request
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        
        # Return the list of responses for the entire batch
        return responses

    def finalize(self):
        """
        Called when the model is unloaded.
        Used to release resources.
        """
        print('Cleaning up YOLO postprocess model...')