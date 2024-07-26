import os
import cv2
from PIL import Image
import numpy as np


def canny_detect(img_path):
    img = cv2.imread(img_path)

    return cv2.Canny(img, 64, 128)


if __name__ == '__main__':
    src_gt = 'E:/Datasets/EORSSD/test-labels'
    dst_a = 'E:/Datasets/EORSSD/test-labels-canny'

    os.makedirs(dst_a, exist_ok=True)

    for img_name in os.listdir(src_gt):
        canny_img = canny_detect(os.path.join(src_gt, img_name))
        cv2.imwrite(os.path.join(dst_a, img_name.replace('.jpg', '.png')), canny_img)