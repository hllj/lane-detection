import numpy as np
import cv2 as cv
from ROI import crop

def convert2HSV(image):
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)

def get_white_mask(image):
    lower_range = (0,0,168)
    upper_range = (172,111,255)
    mask = cv.inRange(image, lower_range, upper_range)
    return mask

def get_yellow_mask(image):
    lower_range = (20, 100, 100)
    upper_range = (30, 255, 255)
    mask = cv.inRange(image, lower_range, upper_range)
    return mask

def get_lane_mask(image):
    hsv = convert2HSV(image)
    yellow_mask = get_yellow_mask(hsv)
    white_mask = get_white_mask(hsv)
    mask = cv.bitwise_or(yellow_mask, white_mask)
    return mask

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
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.Canny(image, 50, 100)
    cv.imshow('Canny', image)
    lines = cv.HoughLinesP(image, 2, np.pi/180, 60, np.array([]), minLineLength=20, maxLineGap=100)
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
