import numpy as np
import cv2
import copy

# Wycinanie i robienie ramki
image = cv2.imread("D:\programy_SI\PROJEKT_SI\znak_1_test.png")
# print(image.shape)
# output = copy.deepcopy(image)
# ball = output[220:300, 120:220]
# output[210:290, 10:110] = ball
# cv2.imshow("images", np.hstack([image, output]))
# cv2.waitKey(0)
#
# constant = cv2.copyMakeBorder(image, 220, 300, 120, 220, cv2.BORDER_CONSTANT, value=[0, 0, 255])
# cv2.imshow("images", constant)
# cv2.waitKey(0)
# # cv.copyMakeBorder()
#
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow("images", hsv)
# cv2.waitKey(0)
#
# # aby wiedziec jakiego koloru szukac w hsv
# red_h = np.uint8([[[0, 0, 255]]])
# hsv_red = cv2.cvtColor(red_h, cv2.COLOR_BGR2HSV)
# print(hsv_red)
#
# red_l = np.uint8([[[0, 0, 150]]])
# hsv_red = cv2.cvtColor(red_l, cv2.COLOR_BGR2HSV)
# print(hsv_red)


image = cv2.medianBlur(image, 5)
#cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=50, param2=30, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
