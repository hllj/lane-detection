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

def get_ROI_perspective(image):
    (height, width, channel) = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    triangle = np.array([[0.1 * width, height], [width // 2, height // 2], [0.9 * width, height]], dtype=np.int32)
    cv.fillPoly(mask, [triangle], 255)
    return mask

def draw_line(image, lines):
    image_lines = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv.line(image_lines, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return image_lines

def get_lane(image):
    mask = get_lane_mask(image)
    masked_color = cv.bitwise_and(image, image, mask=mask)
    masked_ROI = get_ROI_perspective(image)
    ROI = cv.bitwise_and(masked_color, masked_color, mask=masked_ROI)
    ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
    ROI = cv.Canny(ROI, 50, 100)
    cv.imshow('Canny', ROI)
    lines = cv.HoughLinesP(ROI, 2, np.pi/180, 60, np.array([]), minLineLength=30, maxLineGap=5)
    image_lines = draw_line(image.copy(), lines)
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
