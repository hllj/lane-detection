import numpy as np
import cv2 as cv

from ROI import crop
from utils import *
from color_detect import get_lane_mask

def draw_line(image, lines):
    image_lines = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv.line(image_lines, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return image_lines

def get_lane(image):
    image_lines = image.copy()
    image = crop(image)
    cv.imshow('ROI', image)
    mask = get_lane_mask(image)
    image = cv.bitwise_and(image, image, mask=mask)
    cv.imshow('color lane', image)
    image = convert2Gray(image)
    image = cv.Canny(image, 50, 150)
    cv.imshow('Canny', image)
    lines = cv.HoughLinesP(image, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=100)
    image_lines = draw_line(image_lines, lines)
    return image_lines


if __name__ == "__main__":
    image = cv.imread('images/solidYellowCurve.jpg')
    cv.imshow('image', image)
    lane = get_lane(image)
    cv.imshow('lane', lane)
    while True:
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
