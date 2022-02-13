import numpy as np
import cv2
import os
from os import walk
import random
# Wycinanie i robienie ramki
#image = cv2.imread("D:\programy_SI\PROJEKT_SI\znak_1_test.png")
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


# image = cv2.medianBlur(image, 5)
# #cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100,
#                            param1=50, param2=30, minRadius=0, maxRadius=0)
# circles = np.uint16(np.around(circles))
# for i in circles[0, :]:
#     # draw the outer circle
#     cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     # draw the center of the circle
#     cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
# cv2.imshow('detected circles', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#print(colorsys.rgb_to_hsv(168, 18, 33))

#parametry w miare dobre:
#[160, 150, 80]), np.array([180, 255, 255])
# H S V
# H jest od 0 do 180
# S od 0 do 255
# V zostaw od 10 do 255
#ostatni range nie zmieniaj, zostaw od 20 do 255
# H z Gimpa dziel na pół! Bo tam jest zakres 360, w pythonie 180
# S x2.5 z gimpa
#W miarę dobrze:
# np.array([160, 150, 10]), np.array([180, 255, 255])

# np.array([160, 80, 10]), np.array([180, 255, 255])
# path=[]
# for i in range(7):
#     path.append(f"D:\programy_SI\PROJEKT_SI\znak_{i}_test.png")

#testy
#path=[]
# for i in range(900):
#     if os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/noweKolory\/road{i}.png'):
#         path.append(f"D:\programy_SI\PROJEKT_SI\/noweKolory\/road{i}.png")
#
# for path_ in path:
#     image = cv2.imread(path_)
#         # 140 - 165 S30
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask_hsv_lower = cv2.inRange(hsv, np.array([0, 40, 10]), np.array([10, 255, 255]))
#     mask_hsv_upper = cv2.inRange(hsv, np.array([160, 5, 10]), np.array([180, 255, 255]))
#     mask2 = mask_hsv_upper + mask_hsv_lower
#     output_hsv = cv2.bitwise_and(hsv, hsv, mask=mask2)
#
#     # powrot z hsv do RGB i z RGB do szarosci
#     image_RGB = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2BGR)
#     image_HSV_RGB_GREY = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
#     # dobre parametry kółek (341/500 nie znalezionych)
#     image_HSV_RGB_GREY = cv2.medianBlur(image_HSV_RGB_GREY, 5)
#     circles3 = cv2.HoughCircles(image_HSV_RGB_GREY, cv2.HOUGH_GRADIENT, 1, 100,
#                                 param1=50, param2=25, minRadius=0, maxRadius=0)
#     outputHsvDark = None
#     if circles3 is None:
#         mask_hsv_lower = cv2.inRange(hsv, np.array([0, 40, 10]), np.array([10, 255, 255]))
#         dark_mask = cv2.inRange(hsv, np.array([115, 5, 10]), np.array([130, 255, 255]))
#         mask2 = dark_mask + mask_hsv_lower
#         outputHsvDark = cv2.bitwise_and(hsv, hsv, mask=mask2)
#         image_RGB = cv2.cvtColor(outputHsvDark, cv2.COLOR_HSV2BGR)
#         image_HSV_RGB_GREY = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
#         image_HSV_RGB_GREY = cv2.medianBlur(image_HSV_RGB_GREY, 5)
#         circles3 = cv2.HoughCircles(image_HSV_RGB_GREY, cv2.HOUGH_GRADIENT, 1, 100,
#                                     param1=50, param2=25, minRadius=0, maxRadius=0)
#
#     try:
#         circles = np.uint16(np.around(circles3))
#         for i in circles[0, :]:
#             # draw the outer circle
#             cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             # draw the center of the circle
#             cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 3)
#     except:
#         print('Problem z kółkami szarosci')
#
#
#     if outputHsvDark is not None:
#         cv2.imshow(f"images{path_}", np.hstack([image, output_hsv, outputHsvDark]))
#     else:
#         cv2.imshow(f"images{path_}", np.hstack([image, output_hsv]))
#     cv2.waitKey(0)



# czytanie plikow xml
# tree = ET.parse('D:\programy_SI\PROJEKT_SI\/annotations\/road0.xml')
# root = tree.getroot()
# for child in root:
#     print(child.tag, child.attrib)
# print(root[1].text)         #filename
# print(root[4][0].text)      #name np. trafficlight

#
# img = cv2.imread('1.jpg', 1)
# path = 'D:/OpenCV/Scripts/Images'
# cv2.imwrite(os.path.join(path , 'waka.jpg'), img)
# cv2.waitKey(0)
def makingDictionaryForTest():
    arrayWithDicionaries = []
    path = os.getcwd()
    upperPath = os.path.abspath(os.path.join(path, os.pardir))
    mypath = upperPath + "\/test\/images"
    filenames = next(walk(mypath), (None, None, []))[2]
    for object in filenames:
        image = cv2.imread(f'{mypath}\/{object}')
        height, width, channel = image.shape
        arrayWithPartDictionary = []
        imageDictionary = {
            "fileName": object,
            "width": width,
            "height": height,
            "path": upperPath + "\/test",
            "partDictionaries": arrayWithPartDictionary
        }
        arrayWithDicionaries.append(imageDictionary)
    return arrayWithDicionaries

def circleOnImage(dataDict):
    falseCircles = 0
    for data in dataDict:
        path = data["path"] + "\/images\/" + data["fileName"]
        image = cv2.imread(path)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bluredImage = cv2.medianBlur(grey,5)
        circles3 = cv2.HoughCircles(bluredImage, cv2.HOUGH_GRADIENT_ALT, 1, 100,
                                    param1=300, param2=0.85, minRadius=15, maxRadius=150)

        try:
            circles = np.uint16(np.around(circles3))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 3)
                if(i[0] < 1000 and i[1] < 1000 and i[2] < 1000):
                    xmin = 0
                    ymin = 0
                    xmax = i[0] + i[2]
                    ymax = i[1] + i[2]
                    if 0 < (i[0] - i[2]) < int(data["width"]):
                        xmin = i[0] - i[2]
                    if 0 < (i[1] - i[2]) < int(data["height"]):
                        ymin = i[1] - i[2]
                    if xmax > int(data["width"]):
                        xmax = int(data["width"])
                    if ymax > int(data["height"]):
                        ymax = int(data["height"])
                    if xmax - xmin > int(data["width"]) / 10 and ymax - ymin > int(data["height"]) / 10:
                        tmpDictionary = {"xmin": xmin,
                                         "xmax": xmax,
                                         "ymin": ymin,
                                         "ymax": ymax}
                        data["partDictionaries"].append(tmpDictionary)
        except:
            falseCircles += 1
        data.update({"elementsOnImage": len(data["partDictionaries"])})  # ilosc elementow wykrytych poprzez algorytm
        # cv2.imshow("images", image)
        # cv2.waitKey(0)
    print(falseCircles)
    return dataDict

def main():
    dataTest = makingDictionaryForTest()
    print("searching for interesting places on image")
    dataTest = circleOnImage(dataTest)

main()