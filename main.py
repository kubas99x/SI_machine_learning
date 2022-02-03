import numpy as np
import cv2

# load the image
path = []
for i in range(150):
    n =  + i
    #path.append(f"D:\programy_SI\PROJEKT_SI\znak_{n}_test.png")
    path.append(f"D:\programy_SI\PROJEKT_SI\images\/road{n}.png")



for path_ in path:
    image = cv2.imread(path_)
    # mask = cv2.inRange(image, np.array([17, 15, 100]), np.array([50, 56, 200]))
    # output = cv2.bitwise_and(image, image, mask=mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_hsv_lower = cv2.inRange(hsv, np.array([0, 120, 10]), np.array([10, 255, 255]))
    mask_hsv_upper = cv2.inRange(hsv, np.array([160, 80, 10]), np.array([180, 255, 255]))
    #testowa, za dużo wyznacza
    #mask_hsv_upper = cv2.inRange(hsv, np.array([120, 50, 10]), np.array([180, 255, 255]))
    mask2 = mask_hsv_upper + mask_hsv_lower
    output_hsv = cv2.bitwise_and(hsv, hsv, mask=mask2)

    # powrot z hsv do RGB i z RGB do szarosci
    image_RGB = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2BGR)
    image_HSV_RGB_GREY = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
    # dobre parametry kółek
    image_HSV_RGB_GREY = cv2.medianBlur(image_HSV_RGB_GREY, 5)
    circles3 = cv2.HoughCircles(image_HSV_RGB_GREY, cv2.HOUGH_GRADIENT, 1, 100,
                                param1=50, param2=25, minRadius=0, maxRadius=0)
    # testowe parametry kółek
    # image_HSV_RGB_GREY = cv2.medianBlur(image_HSV_RGB_GREY, 5)
    # circles3 = cv2.HoughCircles(image_HSV_RGB_GREY, cv2.HOUGH_GRADIENT_ALT, 1, 20,
    #                            param1=300, param2=0.55, minRadius=0, maxRadius=0)

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

    # cv2.imshow("maska",mask2)
    cv2.imshow(f"images{path_}", np.hstack([image, output_hsv]))
    cv2.waitKey(0)
