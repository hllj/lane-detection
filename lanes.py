import numpy as np
import cv2 as cv

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

if __name__ == "__main__":
    image = cv.imread('images/solidYellowCurve.jpg')
    mask = get_lane_mask(image)
    cv.imshow('lane mask white + yellow', mask)
    cv.imshow('image', image)

    while True:
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
