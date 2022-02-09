import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import os

def loadingImage():
    # load the image
    pathSpeedSights = []
    pathOther = []
    for i in range(900):
        if os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/speedLimitsTrain\/road{i}.png'):
            pathSpeedSights.append(f'D:\programy_SI\PROJEKT_SI\/speedLimitsTrain\/road{i}.png')
        elif os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/otherTrain\/road{i}.png'):
            pathOther.append(f'D:\programy_SI\PROJEKT_SI\/otherTrain\/road{i}.png')
    return pathSpeedSights, pathOther

def makingDictionaryForLearning():
    i=0
    arrayWithDicionaries = []
    while os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml'):
        tree = ET.parse(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml')
        i+=1
        root = tree.getroot()
        fileName = root.find('filename').text
        imageSize = root.find('size')
        if (os.path.isfile(f'D:\programy_SI\PROJEKT_SI\speedLimitsTrain\/{fileName}')) or (os.path.isfile(f'D:\programy_SI\PROJEKT_SI\otherTrain\/{fileName}')) : # checking if image from xml is to learn
            for object in root.findall('object'):
                bndbox = object.find('bndbox')
                bndboxDict = {"xmin": bndbox[0].text,
                               "ymin": bndbox[1].text,
                               "xmax": bndbox[2].text,
                               "ymax": bndbox[3].text}
                imageDictionary = {
                    "filename" : fileName,
                    "width" : imageSize.find('width').text,
                    "height" : imageSize.find('height').text,
                    "name" : object.find('name').text,
                    "bndbox" : bndboxDict
                }
                arrayWithDicionaries.append(imageDictionary)
        else:
            continue
    print(len(arrayWithDicionaries))
    print(arrayWithDicionaries[0]["bndbox"]["xmin"])        #reference to a parameter in dictionary
    return arrayWithDicionaries

def makingMaskForCircles(hsvImage, lowerMaskL, lowerMaskH, higherMaskL, higherMaskH):
    lowerMask = cv2.inRange(hsvImage, lowerMaskL, lowerMaskH)
    higherMask = cv2.inRange(hsvImage, higherMaskL, higherMaskH)
    mask = lowerMask + higherMask
    outputMask = cv2.bitwise_and(hsvImage, hsvImage, mask=mask)  # Mask for HSV
    imageRGB = cv2.cvtColor(outputMask, cv2.COLOR_HSV2BGR)  # HSV to RGB
    imageGREY = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)  # RGB to GRAY
    bluredImage = cv2.medianBlur(imageGREY, 5)  # blur for circles
    return bluredImage

def circleOnImage(path):
    falseCircles = 0;
    for path_ in path:
        circles4 = None
        circles5 = None
        image = cv2.imread(path_)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        bluredImage1 = makingMaskForCircles(hsv, np.array([0, 120, 10]), np.array([10, 255, 255]),
                                            np.array([160, 80, 10]), np.array([180, 255, 255]))
        circles3 = cv2.HoughCircles(bluredImage1, cv2.HOUGH_GRADIENT, 1, 100,
                                    param1=50, param2=25, minRadius=0, maxRadius=0)
        if circles3 is None:
            bluredImage2 = makingMaskForCircles(hsv, np.array([0, 40, 10]), np.array([10, 255, 255]),
                                                np.array([140, 5, 10]), np.array([180, 255, 255]))
            circles4 = cv2.HoughCircles(bluredImage2, cv2.HOUGH_GRADIENT, 1, 100,
                                        param1=50, param2=25, minRadius=0, maxRadius=0)
        if (circles3 is None) & (circles4 is None):
            bluredImage3 = makingMaskForCircles(hsv, np.array([0, 40, 10]), np.array([10, 255, 255]),
                                                np.array([115, 5, 10]), np.array([130, 255, 255]))
            circles5 = cv2.HoughCircles(bluredImage3, cv2.HOUGH_GRADIENT, 1, 100,
                                        param1=50, param2=25, minRadius=0, maxRadius=0)
        circles = None
        if circles3 is not None:
            circles = np.uint16(np.around(circles3))
        elif circles4 is not None:
            circles = np.uint16(np.around(circles4))
        elif circles5 is not None:
            circles = np.uint16(np.around(circles5))
        try:
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 3)
        except:
            falseCircles += 1

        # cv2.imshow(f"images{path_}", np.hstack([image]))
        # cv2.waitKey(0)
    print(falseCircles)

def main():
    #pathsWithSpeedSights, pathsWithOther = loadingImage()
    #circleOnImage(pathsWithSpeedSights)
    arrayDictionaryWithImageParameters = makingDictionaryForLearning()

if __name__ == '__main__':
    main()
