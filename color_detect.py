import numpy as np
import cv2 as cv

from utils import *

def get_white_mask(image):
    '''
        detect white lane in grayscale image, use white range is 200 - 255
    '''
    mask = cv.inRange(image, 200, 255)
    return mask

def get_yellow_mask(image):
    '''
    detect yellow lane in image with hsv color space
    '''
    lower_range = (20, 100, 100)
    upper_range = (30, 255, 255)
    mask = cv.inRange(image, lower_range, upper_range)
    return mask

def get_lane_mask(image):
    hsv = convert2HSV(image)
    grayscale = convert2Gray(image)
    yellow_mask = get_yellow_mask(hsv)
    white_mask = get_white_mask(grayscale)
    mask = cv.bitwise_or(yellow_mask, white_mask)
    return mask

