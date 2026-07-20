import argparse
import numpy as np
import sys
import gevent.ssl
import cv2

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def test_infer(model_name, input0_data ):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('image', [3, 1024, 1024], "FP32"))
    
    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    
    outputs.append(httpclient.InferRequestedOutput('6568', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('6570', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('6572', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('6887', binary_data=False))


    results = triton_client.infer(model_name,
                                  inputs,
                                  outputs=outputs)

    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable encrypted link to the server using HTTPS')

    FLAGS = parser.parse_args()
    try:
        if FLAGS.ssl:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url,
                verbose=FLAGS.verbose,
                ssl=True,
                ssl_context_factory=gevent.ssl._create_unverified_context,
                insecure=True)
        else:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "maskrcnn_onnx"

    # Create the data for the input tensors. With the image in CHW.
    image = cv2.imread("person_dog.jpg")
    image_np = np.transpose( image, [2, 0, 1] ) # CHW
    input0_data = image_np.astype(np.float32) # FP32
    

    # Infer with requested Outputs
    results = test_infer(model_name, input0_data )
    
    # Validate the results by comparing with precomputed values.
    boxes = results.as_numpy('6568') # boxes
    labels = results.as_numpy('6570') # labels
    scores = results.as_numpy('6572') # scores
    masks = results.as_numpy('6887') # masks

    
    for mask, box, label, score in zip(masks, boxes, labels, scores):
        # Showing boxes with score > 0.7
        if score <= 0.7:
            continue

        # Label
        print(score, " " , label )
        color = (255, 0, 0) 

        # Bounding Box
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])        
        thickness = 2

        image_dt = cv2.rectangle(image, start_point, end_point, color, thickness) 

        
        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, image.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, image.shape[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[ mask_y_0 : mask_y_1, mask_x_0 : mask_x_1 ]
        im_mask = im_mask[:, :, None]

        contours, hierarchy = cv2.findContours( im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        image_dt = cv2.drawContours(image_dt, contours, -1, color, 3)
     



    cv2.imwrite( "person_dog_detected.jpg",image_dt)

    print("PASS!")