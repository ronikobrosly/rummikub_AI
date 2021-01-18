
from os.path import expanduser

import cv2
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np


# Create kernel for cv2 dilation method
KERNEL = np.ones((5,5),np.uint8)


# Import the model
model = load_model('big_model')


# Read input image
img = cv2.imread(expanduser('~/Desktop/rummikub/images/prediction_test/pred_pic.png'))

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imgray, (5, 5), 0)

edges = cv2.Canny(blurred, 100, 250)
edges = cv2.dilate(edges, KERNEL, iterations = 1)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)

for points in contours[0]:
    coor_list = points[0].tolist()
    edges = cv2.circle(edges, (coor_list[0],coor_list[1]), radius=5, color=(0, 250, 0), thickness=5)

cv2.imshow('edges', edges)
cv2.destroyAllWindows()



# Helpful links to continue this:

# https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/
# https://www.youtube.com/watch?v=6DjFscX4I_c
# https://stackoverflow.com/questions/60873721/python-contour-around-rectangle-based-on-specific-color-on-a-dark-image-opencv
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# https://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python/
