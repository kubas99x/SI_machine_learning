import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas



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
    i = 0
    arrayWithDicionaries = []
    badDimensions = 0
    while os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml'):
        tree = ET.parse(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml')
        i += 1
        root = tree.getroot()
        fileName = root.find('filename').text
        imageSize = root.find('size')
        if (os.path.isfile(f'D:\programy_SI\PROJEKT_SI\speedLimitsTrain\/{fileName}')) or (
                os.path.isfile(
                    f'D:\programy_SI\PROJEKT_SI\otherTrain\/{fileName}')):  # checking if image from xml is to learn
            for ob in root.findall('object'):
                bndbox = ob.find('bndbox')
                bndboxDict = {"xmin": int(bndbox[0].text),
                              "ymin": int(bndbox[1].text),
                              "xmax": int(bndbox[2].text),
                              "ymax": int(bndbox[3].text)}
                xsize = int(bndboxDict["xmax"]) - int(bndboxDict["xmin"])
                ysize = int(bndboxDict["ymax"]) - int(bndboxDict["ymin"])
                if ((xsize < int(imageSize.find('width').text) / 10) or (
                        ysize < int(imageSize.find('height').text) / 10)):
                    badDimensions += 1
                    continue
                else:
                    status = None
                    if ob.find('name').text == "speedlimit":
                        status = 1
                    else:
                        status = 2
                    imageDictionary = {
                        "fileName": fileName,
                        "width": imageSize.find('width').text,
                        "height": imageSize.find('height').text,
                        "name": ob.find('name').text,
                        "bndbox": bndboxDict,
                        "status": status  # 1 if speedsight, 2 if other
                    }
                    arrayWithDicionaries.append(imageDictionary)
        else:
            continue
    print(len(arrayWithDicionaries))
    print(arrayWithDicionaries[0]["bndbox"]["xmin"])  # reference to a parameter in dictionary
    print("ilosc odrzuconych wycinkow: ", badDimensions)
    return arrayWithDicionaries


def learningBOW(imageDictionary):
    dictionarySize = 4
    bow = cv2.BOWKMeansTrainer(dictionarySize)
    sift = cv2.SIFT_create()
    for part in imageDictionary:
        if os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/images\/{part["fileName"]}'):
            image = cv2.imread(f'D:\programy_SI\PROJEKT_SI\/images\/{part["fileName"]}')
            bndbox = part["bndbox"]
            sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
            try:
                grey = cv2.cvtColor(sightPart, cv2.COLOR_BGR2GRAY)
                kp = sift.detect(grey, None)
                kp, desc = sift.compute(grey, kp)
                if desc is not None:
                    bow.add(desc)
            except:
                print("error in sift")
    dictionary = bow.cluster()
    np.save('dict.npy', dictionary)  # zapisanie naszego slownika do pliku


def extract(imageDictionary):
    # SIFT jest algorytmem do wyznaczania wlasciwosci danego zdjecia
    # FlannBasedMatcher sluzy do znajdowania najblizszych matchy dla parametrow sift
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()  # finds the best matches
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)  # descriptor extractor, desriptor matcher
    dictionary = np.load('dict.npy')  # size 2x128
    bow.setVocabulary(dictionary)
    for part in imageDictionary:
        if os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/images\/{part["fileName"]}'):
            image = cv2.imread(f'D:\programy_SI\PROJEKT_SI\/images\/{part["fileName"]}')
            bndbox = part["bndbox"]
            sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
            grey = cv2.cvtColor(sightPart, cv2.COLOR_BGR2GRAY)
            desc = bow.compute(grey, sift.detect(grey))  # input KeypointDescriptor, output imgDescriptor
            # print("extract desc: ", desc)
            if desc is not None:
                part.update({'desc': desc})
            else:
                part.update({'desc': np.zeros((1, len(dictionary)))})
    return imageDictionary


def train(imageDictionary):
    print("LETS GO WITH TRAINING!")
    clf = RandomForestClassifier(100)
    y_train = []
    x_train = np.empty((1, 4))
    for sample in imageDictionary:
        y_train.append(sample["status"])
        x_train = np.vstack((x_train, sample["desc"]))
    clf.fit(x_train[1:], y_train)
    return clf


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
    pathsWithSpeedSights, pathsWithOther = loadingImage()
    # summedPaths = pathsWithSpeedSights + pathsWithOther
    # circleOnImage(pathsWithSpeedSights)
    dictWithImgPar = makingDictionaryForLearning()
    learningBOW(dictWithImgPar)
    dictWithImgPar = extract(dictWithImgPar)  # dictionary with added descriptor parameters

    # print('dictionary: ', dictWithImgPar[0])
    # for n in dictWithImgPar:
    #     print(n["name"], n["desc"])
    afterTrain = train(dictWithImgPar)


    # print("Predict: ", afterTrain.predict([[0.2, 0.3, 0.1, 0.4]]))
    # print("Predict2: ", afterTrain.predict([[0.25, 0.05, 0.53, 0.17]]))
    # print("Predict3: ", afterTrain.predict([[0.2857143,  0.07142857, 0.2857143,  0.35714287]]))

if __name__ == '__main__':
    main()
