import math, cv2
import numpy as np

def resize_norm_img(img, return_shape = (3, 48, 320)) :
    imgC, imgH, imgW = return_shape
    max_wh_ratio = imgW / imgH
    imgW = int((imgH * max_wh_ratio))

    h, w = img.shape[:2]
    ratio = w / float(h)

    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))

    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    
    return padding_im
