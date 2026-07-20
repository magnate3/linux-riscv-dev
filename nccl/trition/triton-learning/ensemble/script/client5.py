import tritonclient.http  as httpclient
from tritonclient.utils import InferenceServerException


import matplotlib.pyplot as plt
import numpy as np
import cv2
np.bool = np.bool_

from operators.detection_preprocess import resize, normalize
from operators.recognition_preprocess import resize_norm_img
from operators.recognition_postprocess import CTCLabelDecode


import logging
from PIL import Image
from io import BytesIO

logging.basicConfig(level = logging.DEBUG)

# client 정의
triton_client = httpclient.InferenceServerClient(
    url = 'localhost:8000',
    verbose = False

)



def test_infer(model_name, input_data, input_name, input_type, output_tensor_name, headers = None ) :
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput(input_name, input_data.shape, input_type))

    inputs[0].set_data_from_numpy(input_data, binary_data = False)

    #outputs.append(httpclient.InferRequestedOutput(output_tensor_name, binary_data = False))

    results = triton_client.infer(
        model_name,
        inputs,
        #outputs = outputs,
        headers = headers,
    )

    return results

def test_preprocessing(model_name, 
                       input_data, 
                       input_name, 
                       input_type, 
                       output1_tensor_name, # 변수 갯수를 포괄하는 일반화 필요 e.g) output1, output2, ...
                       output2_tensor_name,
                       headers = None ) :
    '''detection preprocessing has two outputs : ['detection_preprocessing_output', 'original_image_info'] '''
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput(input_name, input_data.shape, input_type))

    inputs[0].set_data_from_numpy(input_data, binary_data = False)

    outputs = [
        httpclient.InferRequestedOutput(output1_tensor_name, binary_data = False),
        httpclient.InferRequestedOutput(output2_tensor_name, binary_data = False)
    ]
    
    results = triton_client.infer(
        model_name,
        inputs,
        outputs = outputs,
        headers = headers,
    )

    return results




if __name__ == '__main__' :
    TEST_TYPE = 'text_recognition'
    TEST_TYPE = 'text_detection'
    logging.debug(f'test type : {TEST_TYPE}')
    
    
    if TEST_TYPE == 'text_detection' : 
        model_name = 'ensemble_model'
        output_tensor_name = 'sigmoid_0.tmp_0'
        # sample image
        #input_data = plt.imread('test_image/detection_sample_image.jpg')
        input_data = plt.imread('input.jpg')
        

        
        # paddleocr detection preprocessing 1) DetResizeForText, NormalizeImage, ToCHWImage
        img, (src_h, src_w, ratio_h, ratio_w) = resize(input_data)
        logging.debug(f'original image size(h,w) : ({src_h}, {src_w}), ratio(h,w) : ({ratio_h}, {ratio_w})')
        logging.debug(f'after resize : {img.shape}' )

        norm_img = normalize(img)
        transpose_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #transpose_img = np.transpose(norm_img, (2,0,1))

        #transpose_img = np.expand_dims(transpose_img.astype(dtype = np.uint8), 0)
        logging.debug(f'input image type : [{type(transpose_img)}, shape : {transpose_img.shape}]')

    

        result = test_infer(model_name, 
                            input_data = transpose_img, 
                            input_name = 'input_image',
                            input_type = 'UINT8',
                            output_tensor_name = output_tensor_name)
        print(result.as_numpy('sigmoid_0.tmp_0'))
        # img = Image.open(img_data)

        plt.imshow(result.as_numpy('sigmoid_0.tmp_0')[0,0,:,:])
        plt.show()

        np.save('test_image/featuremap.npy', result.as_numpy('sigmoid_0.tmp_0'))

    elif TEST_TYPE == 'text_recognition' :

        model_name = 'text_recognition'
        output_tensor_name = 'softmax_5.tmp_0'

        # # sample image
        # input_data = cv2.imread('test_image/recognition_sample_image2.jpg')

        # # padddleocr recognition preprocessing 
        # resized_img = np.expand_dims(resize_norm_img(input_data), 0)
        # print(resized_img)
        # plt.imshow(resized_img[0].transpose((1,2,0)))
        # plt.show()

        recognition_input = np.load('test_image/recognition_input.npy').astype('float32')

        # send request to triton
        result = test_infer(model_name, 
                            input_name = 'x',
                            input_data = recognition_input, 
                            input_type = 'FP32', 
                            output_tensor_name = output_tensor_name)

        # post-processing
        ctclabeldecoder = CTCLabelDecode(character_dict_path = 'assets/koreadeep_char_dict_unicode.txt',
                                         use_space_char = True
                                         )

        logging.debug(f'output shape : {result.as_numpy(output_tensor_name).shape}')
        
        
        decoded_result = ctclabeldecoder(result.as_numpy(output_tensor_name))

        print(decoded_result)

    elif TEST_TYPE == 'detection_preprocessing' :
        
        model_name = 'detection_preprocessing'
        output1_tensor_name = 'detection_preprocessing_output'
        output2_tensor_name = 'original_image_info'

        image_data = np.fromfile("test_image/detection_sample_image.jpg", dtype="uint8")
        image_data = np.expand_dims(image_data, axis=0)
        logging.debug(f'image input shape : {image_data.shape}')
        response = test_preprocessing(model_name, 
                            input_data = image_data,  
                            input_name = 'detection_preprocessing_input',
                            input_type = 'UINT8',
                            output1_tensor_name = output1_tensor_name,
                            output2_tensor_name = output2_tensor_name
                            )

        # response = test_infer(model_name, 
        #                     input_name = 'detection_preprocessing_input',
        #                     input_data = image_data, 
        #                     input_type = 'UINT8', 
        #                     output_tensor_name = output1_tensor_name)


        result = response.get_response()

        logging.debug(f'output shape : {response.as_numpy(output1_tensor_name).shape}')
        logging.debug(f'output2 : {response.as_numpy(output2_tensor_name)}')

        plt.imshow(np.transpose(response.as_numpy(output1_tensor_name), (1,2,0)))
        plt.show()

    elif TEST_TYPE == 'detection_postprocessing' :
        model_name = 'detection_postprocessing'
        output_tensor_name = 'detection_postprocessing_output'

        original_image_data = np.fromfile("test_image/detection_sample_image.jpg", dtype="uint8")
        original_image_data = np.expand_dims(original_image_data, axis=0)

        # get heatmap result from detection result
        featuremap = np.load('test_image/featuremap.npy').astype('float32')

        print(featuremap.shape)
        h,w = (2368, 1728)
        ratio_h, ratio_w = (0.40540540540540543, 0.4074074074074074)

        shape_list = np.array([[h, w, ratio_h, ratio_w]], dtype = np.float32)


        # need original image information to crop the original image

        
        outputs = []
        inputs = [
            httpclient.InferInput('detection_postprocessing_input', featuremap.shape, 'FP32'),
            httpclient.InferInput('original_image', original_image_data.shape, 'UINT8'),
            httpclient.InferInput('original_image_info', shape_list.shape, 'FP32'),
        ]
        

        inputs[0].set_data_from_numpy(featuremap, binary_data = False) # featuremap
        inputs[1].set_data_from_numpy(original_image_data, binary_data = False) # original image info
        inputs[2].set_data_from_numpy(shape_list, binary_data = False) # original image

        outputs = [
            httpclient.InferRequestedOutput(output_tensor_name, binary_data = False),
        ]
        
        results = triton_client.infer(
            model_name,
            inputs,
            outputs = outputs,
        )

        # expect cropped images for STR input

        cropped_images = results.as_numpy(output_tensor_name)

        np.save('test_image/recognition_input.npy', cropped_images)

        logging.debug(f'cropped images shape : {cropped_images.shape}')

        # plt.imshow(np.transpose(cropped_images[1], (1,2,0)))
        # plt.show()
        


        
    elif TEST_TYPE == 'recognition_postprocessing' :
        model_name = 'recognition_postprocessing'
        output_tensor_name = 'recognition_postprocessing_output'

