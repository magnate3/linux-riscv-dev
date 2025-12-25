# Copyright (C) 2025 YIQISOFT
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse  # Import argparse for command-line arguments

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

# YOLOv8n is typically trained on the COCO 80 classes.
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def preprocess(image, input_size=(640, 640)):
    """
    Preprocesses the image: resize, normalize, HWC -> CHW, and add batch dimension (Letterbox).
    """
    # 1. Get original dimensions
    h, w, _ = image.shape
    
    # 2. Calculate scaling factor and resize
    scale = min(input_size[0] / h, input_size[1] / w)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Create a canvas padded with gray (114)
    padded_image = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    
    # 4. Paste the resized image onto the center of the canvas
    dw, dh = (input_size[1] - resized_w) // 2, (input_size[0] - resized_h) // 2
    padded_image[dh:dh + resized_h, dw:dw + resized_w, :] = resized_image
    
    # 5. Normalize (0-255 -> 0.0-1.0)
    padded_image = padded_image.astype(np.float32) / 255.0
    
    # 6. HWC -> CHW
    padded_image = padded_image.transpose((2, 0, 1))
    
    # 7. Add batch dimension
    padded_image = np.expand_dims(padded_image, axis=0)
    
    return padded_image, scale, (dw, dh)

def postprocess_and_draw(image, results, scale, pad_wh):
    """
    Draws bounding boxes on the original image.
    results: Formatted output data received from Triton.
    """
    # Get padding dimensions
    pad_w, pad_h = pad_wh
    
    for box in results:
        # box: [image_id, label_id, conf, xmin, ymin, xmax, ymax]
        image_id, label_id_int, conf, xmin, ymin, xmax, ymax = box

        # Look up the class name
        label_id = int(label_id_int) # Ensure it's an integer index
        label_name = CLASSES[label_id] if label_id < len(CLASSES) else f"Unknown({label_id})"
        
        # Rescale coordinates back to original image dimensions
        xmin = (xmin - pad_w) / scale
        ymin = (ymin - pad_h) / scale
        xmax = (xmax - pad_w) / scale
        ymax = (ymax - pad_h) / scale
        
        # Draw the rectangle (color is green)
        color = (0, 255, 0)
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        
        # Prepare label text
        label_text = f"{label_name}: {conf:.2f}"
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (int(xmin), int(ymin) - text_h - 5), (int(xmin) + text_w, int(ymin)), color, -1)
        
        # Draw label text
        cv2.putText(image, label_text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return image

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Triton gRPC client for YOLOv8 ensemble model.")
    parser.add_argument('--url', type=str, required=True, 
                        help='Triton server URL (e.g., 192.168.200.122:8001 for gRPC).')
    parser.add_argument('--image', type=str, default='input.jpg', 
                        help='Path to the input image file (default: input.jpg).')
    
    args = parser.parse_args()
    
    TRITON_URL = args.url
    image_path = args.image

    # --- 1. Initialize Triton Client (using gRPC) ---
    try:
        # Initialize gRPC client using the provided URL
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL) 
    except Exception as e:
        print(f"Failed to create Triton gRPC client at {TRITON_URL}: {e}")
        exit(1)

    # --- 2. Prepare Input Data ---
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: could not read image at {image_path}")
        exit(1)

    # Preprocess
    input_image, scale, pad_wh = preprocess(original_image)
    
    # --- 3. Create Triton Input Tensor (using gRPC API) ---
    input_tensor = grpcclient.InferInput('image_input', input_image.shape, "FP32")
    input_tensor.set_data_from_numpy(input_image)

    # --- 4. Call Triton Service ---
    model_name = "yolov8n_ensemble"
    print(f"Invoking model '{model_name}' using gRPC...")
    
    try:
        # Perform inference
        results = triton_client.infer(model_name=model_name, inputs=[input_tensor])
    except Exception as e:
        print(f"gRPC Inference failed: {e}")
        exit(1)
        
    # --- 5. Parse and Process Results ---
    # Retrieve the final_boxes output tensor
    output_boxes = results.as_numpy('final_boxes')
    
    if output_boxes.size == 0:
        print("No objects detected.")
    else:
        # Print results for the first detected box (including label name)
        first_box = output_boxes[0]
        label_id = int(first_box[1])
        label_name = CLASSES[label_id] if label_id < len(CLASSES) else f"Unknown({label_id})"

        print(f"Detected {len(output_boxes)} objects.")
        print(f"First detection: Label='{label_name}' (ID={label_id}), Conf={first_box[2]:.2f}")

        # --- 6. Draw Bounding Boxes on Image ---
        result_image = postprocess_and_draw(original_image.copy(), output_boxes, scale, pad_wh)

        # Save the result
        output_image_path = "output_detection.jpg"
        cv2.imwrite(output_image_path, result_image)
        print(f"Result image saved to {output_image_path}")
