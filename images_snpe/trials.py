import cv2
import numpy as np

image1 = cv2.imread('chairs.jpg')
image2 = cv2.imread('handicap_sign.jpg')
image3 = cv2.imread('notice_sign.jpg')
image4 = cv2.imread('plastic_cup.jpg')
image5 = cv2.imread('trash_bin.jpg')

# for i in range(1, 6, 1):
#     file = "image" + str(i)
#     filename = file + ".raw"

np.ndarray.tofile(image1, 'img1.raw')
np.ndarray.tofile(image2, 'img2.raw')
np.ndarray.tofile(image3, 'img3.raw')
np.ndarray.tofile(image4, 'img4.raw')
np.ndarray.tofile(image5, 'img5.raw')
