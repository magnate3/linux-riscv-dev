import cv2
import sys
from PIL import Image
import numpy as np

def resize(img, limit_side_len = 960, limit_type_length = 'max' ) :
    '''
    params : limit_side_len, limit_type_length 고정
    추가 정보는 ppocr/data/imaug/operators.py 참고
    '''
    
    h, w, _ = img.shape
    src_h, src_w = h, w
    if max(h, w) > limit_side_len:
        if h > w:
            ratio = float(limit_side_len) / h
        else:
            ratio = float(limit_side_len) / w
    else:
        ratio = 1.

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            return None, (None, None)
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
    except:
        print(img.shape, resize_w, resize_h)
        sys.exit(0)
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    

    return img, (src_h, src_w, ratio_h, ratio_w)


def normalize(img, 
              scale = 1. / 255., 
              mean = [0.485, 0.456, 0.406],
              std = [0.229, 0.224, 0.225]
              ) :
    if isinstance(img, Image.Image):
        img = np.array(img)
    assert isinstance(img,
                        np.ndarray), "invalid input 'img' in NormalizeImage"
    img = (img.astype('float32') * scale - mean) / std

    return img


