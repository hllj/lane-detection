import numpy as np
import cv2 as cv

def convert2HSV(image):
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)
def convert2Gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


