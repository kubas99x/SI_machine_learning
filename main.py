import numpy as np
import cv2



# define the list of boundaries
# B G R
# boundaries = [
#     ([17, 15, 100], [50, 56, 200]),
#     ([86, 31, 4], [220, 88, 50]),
#     ([25, 146, 190], [62, 174, 250]),
#     ([103, 86, 65], [145, 133, 128])
# ]
# loop over the boundaries
# for (lower, upper) in boundaries:
#     # create NumPy arrays from the boundaries
#     lower = np.array(lower, dtype="uint8")
#     upper = np.array(upper, dtype="uint8")
#     # find the colors within the specified boundaries and apply the mask
#     mask = cv2.inRange(image, lower, upper)
#     output = cv2.bitwise_and(image, image, mask=mask)
#     # show the images
#     cv2.imshow("images", np.hstack([image, output]))
#     cv2.waitKey(0)

# load the image
path=[]
for i in range(7):
    path.append(f"D:\programy_SI\PROJEKT_SI\znak_{i}_test.png")

#image = cv2.imread("D:\programy_SI\PROJEKT_SI\znak_1_test.png")

for path_ in path:
    image = cv2.imread(path_)
    mask = cv2.inRange(image, np.array([17, 15, 100]), np.array([50, 56, 200]))
    output = cv2.bitwise_and(image, image, mask=mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([160, 150, 80]), np.array([180, 255, 255]))
    output_hsv = cv2.bitwise_and(hsv, hsv, mask=mask2)

    output_circle_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles2 = cv2.HoughCircles(output_circle_2, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=25, minRadius=0, maxRadius=0)

    # powrot z hsv do RGB i z RGB do szarosci
    image_RGB = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2BGR)
    image_HSV_RGB_GREY = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
    circles3 = cv2.HoughCircles(image_HSV_RGB_GREY, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=25, minRadius=0, maxRadius=0)

    try:
        circles = np.uint16(np.around(circles2))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output_circle_2, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output_circle_2, (i[0], i[1]), 2, (0, 0, 255), 3)
    except:
        print('Problem z kółkami na RGB')

    # dla próby malowanie kółek na image
    try:
        circles = np.uint16(np.around(circles3))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 3)
    except:
        print('Problem z kółkami szarosci')

    #cv2.imshow("maska",mask2)
    cv2.imshow("images", np.hstack([image, output, output_hsv]))
    cv2.imshow("circles in red", output_circle_2)
    cv2.waitKey(0)

