import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
from sklearn.ensemble import RandomForestClassifier
from os import walk
import random

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.953 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0


# TODO Skuteczność detekcji mAP = 0.371 (4/6) (1.5/2)

# TODO max(4, 4+1.5) = 5.5


def makingDictionaryForLearning():
    arrayWithDicionaries = []
    path = os.getcwd()
    upperPath = os.path.abspath(os.path.join(path, os.pardir))
    mypath = upperPath + "/train/annotations"
    filenames = next(walk(mypath), (None, None, []))[2]
    for object in filenames:
        tree = ET.parse(f'{mypath}/{object}')
        root = tree.getroot()
        imageSize = root.find('size')
        partDictionaryArray = []
        for ob in root.findall('object'):
            bndbox = ob.find('bndbox')
            xmin = int(bndbox[0].text)
            ymin = int(bndbox[1].text)
            xmax = int(bndbox[2].text)
            ymax = int(bndbox[3].text)
            xsize = xmax - xmin
            ysize = ymax - ymin
            if ((xsize < int(imageSize.find('width').text) / 10) or (
                    ysize < int(imageSize.find('height').text) / 10)):
                continue
            else:
                if ob.find('name').text == "speedlimit":
                    status = 1
                else:
                    status = 2
                partDicionary = {
                    "name": ob.find('name').text,
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "status": status
                }
                partDictionaryArray.append(partDicionary)
        if not partDictionaryArray:  # jezeli wszystkie wycinki w obrazie byly za male, pomijamy to zdjecie
            continue
        else:
            imageDictionary = {
                "fileName": root.find('filename').text,
                "width": imageSize.find('width').text,
                "height": imageSize.find('height').text,
                "path": upperPath + "/train",
                "partDictionaries": partDictionaryArray
            }
            arrayWithDicionaries.append(imageDictionary)
    return arrayWithDicionaries


def addingPartsToTrainData(dataTrain):
    for object in dataTrain:
        if len(object["partDictionaries"]) == 1 and object["partDictionaries"][0]["status"] == 2:
            for n in range(4):
                # TODO Moznaby dodac rozne wielkosci wycinkow.
                xmin = random.randint(0, int(object["width"]) - 55)
                xmax = xmin + 50
                ymin = random.randint(0, int(object["height"]) - 55)
                ymax = ymin + 50
                tmpDictionary = {
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "status": 2
                }
                object["partDictionaries"].append(tmpDictionary)
        elif len(object["partDictionaries"]) == 1 and object["partDictionaries"][0]["status"] == 1:
            for n in range(2):
                xminOryginal = object["partDictionaries"][0]["xmin"]
                xmaxOryginal = object["partDictionaries"][0]["xmax"]
                yminOryginal = object["partDictionaries"][0]["ymin"]
                ymaxOryginal = object["partDictionaries"][0]["ymax"]
                # liczby, by iou nie było mniejsze niż 50%
                number = random.randrange(-15, 15) / 100
                xmin = xminOryginal - number * (xmaxOryginal - xminOryginal)
                xmax = xmaxOryginal - number * (xmaxOryginal - xminOryginal)
                number = random.randrange(-15, 15) / 100
                ymin = yminOryginal - number * (ymaxOryginal - yminOryginal)
                ymax = ymaxOryginal - number * (ymaxOryginal - yminOryginal)
                if (xmin < 0) or (xmax > int(object["width"])):
                    continue
                if (ymin < 0) or (ymax > int(object["height"])):
                    continue
                tmpDictionary = {
                    "xmin": int(xmin),
                    "xmax": int(xmax),
                    "ymin": int(ymin),
                    "ymax": int(ymax),
                    "status": 1
                }
                object["partDictionaries"].append(tmpDictionary)
    return dataTrain


def makingDictionaryForTest():
    arrayWithDicionaries = []
    path = os.getcwd()
    upperPath = os.path.abspath(os.path.join(path, os.pardir))
    mypath = upperPath + "/test/images"
    filenames = next(walk(mypath), (None, None, []))[2]
    for object in filenames:
        image = cv2.imread(f'{mypath}/{object}')
        height, width, channel = image.shape
        arrayWithPartDictionary = []
        imageDictionary = {
            "fileName": object,
            "width": width,
            "height": height,
            "path": upperPath + "/test",
            "partDictionaries": arrayWithPartDictionary
        }
        arrayWithDicionaries.append(imageDictionary)
    return arrayWithDicionaries


def learningBOW(imageDictionary):
    dictionarySize = 100
    bow = cv2.BOWKMeansTrainer(dictionarySize)
    sift = cv2.SIFT_create()
    for part in imageDictionary:
        image = cv2.imread(f'{part["path"]}/images/{part["fileName"]}')
        for object in part["partDictionaries"]:
            sightPart = image[object["ymin"]:object["ymax"], object["xmin"]:object["xmax"]]
            try:
                grey = cv2.cvtColor(sightPart, cv2.COLOR_BGR2GRAY)
                kp = sift.detect(grey, None)
                kp, desc = sift.compute(grey, kp)
                if desc is not None:
                    bow.add(desc)
            except:
                pass
    dictionary = bow.cluster()
    np.save('dict.npy', dictionary)  # zapisanie naszego slownika do pliku


def extract(imageDictionary):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()  # finds the best matches
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)  # descriptor extractor, desriptor matcher
    dictionary = np.load('dict.npy')
    bow.setVocabulary(dictionary)
    for part in imageDictionary:
        image = cv2.imread(f'{part["path"]}/images/{part["fileName"]}')
        if len(part["partDictionaries"]) != 0:  # it could be when we are taking parts from circles
            for object in part["partDictionaries"]:
                sightPart = image[object["ymin"]:object["ymax"], object["xmin"]:object["xmax"]]
                grey = cv2.cvtColor(sightPart, cv2.COLOR_BGR2GRAY)
                desc = bow.compute(grey, sift.detect(grey))  # input KeypointDescriptor, output imgDescriptor
                if desc is not None:
                    object.update({'desc': desc})
                else:
                    object.update({'desc': np.zeros((1, len(dictionary)))})
    return imageDictionary


def train(imageDictionary):
    clf = RandomForestClassifier(100)
    y_train = []
    x_train = np.empty((1, 100))
    for sample in imageDictionary:
        for object in sample["partDictionaries"]:
            y_train.append(object["status"])
            x_train = np.vstack((x_train, object["desc"]))
    clf.fit(x_train[1:], y_train)
    return clf


def predictImage(rf, data):
    for sample in data:
        for object in sample["partDictionaries"]:
            object.update({"predictedStatus": rf.predict(object['desc'])})
    return data


def printingChosenparts(dataTest):
    for data in dataTest:
        image = cv2.imread(f'{data["path"]}/images/{data["fileName"]}')
        for part in data["partDictionaries"]:
            if (part["predictedStatus"] == 1):
                image = cv2.rectangle(image, (part['xmin'], part['ymin']), (part['xmax'], part['ymax']),
                                      (0, 255, 0), 1)
            else:
                image = cv2.rectangle(image, (part['xmin'], part['ymin']), (part['xmax'], part['ymax']),
                                      (0, 0, 255), 1)
        cv2.imshow("images", image)
        cv2.waitKey(0)


def detectInformation(dataTest):
    iloscZnakow = 0
    for data in dataTest:
        print(data["fileName"])
        if data["partDictionaries"]:
            foundSights = 0
            for part in data["partDictionaries"]:
                if part["predictedStatus"] == 1:
                    foundSights += 1
            print(foundSights)
        else:
            print('0')
        if data["partDictionaries"]:
            for part in data["partDictionaries"]:
                if part["predictedStatus"] == 1:
                    print(part["xmin"], part["xmax"], part["ymin"], part["ymax"], )
                    iloscZnakow += 1


def classifyInput():
    upperPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    mypath = upperPath + "/" + "test"
    dataClassify = []
    numberOfFiles = input()
    for number in range(int(numberOfFiles)):
        filename = input()
        numberOfParts = input()
        partDictionaryArray = []
        classifyDict = {"fileName": filename,
                        "path": mypath,
                        "partDictionaries": partDictionaryArray}
        for n in range(int(numberOfParts)):
            xmin, xmax, ymin, ymax = input().split()
            tmpDictionary = {"xmin": int(xmin),
                             "xmax": int(xmax),
                             "ymin": int(ymin),
                             "ymax": int(ymax)}
            classifyDict["partDictionaries"].append(tmpDictionary)
        dataClassify.append(classifyDict)
    return dataClassify


def classifyReturn(dataTest):
    for data in dataTest:
        for part in data["partDictionaries"]:
            if part["predictedStatus"] == 1:
                print("speedlimit")
            else:
                print("other")


def makingMaskForCircles(hsvImage, lowerMaskL, lowerMaskH, higherMaskL, higherMaskH):
    lowerMask = cv2.inRange(hsvImage, lowerMaskL, lowerMaskH)
    higherMask = cv2.inRange(hsvImage, higherMaskL, higherMaskH)
    mask = lowerMask + higherMask
    outputMask = cv2.bitwise_and(hsvImage, hsvImage, mask=mask)  # Mask for HSV
    imageRGB = cv2.cvtColor(outputMask, cv2.COLOR_HSV2BGR)  # HSV to RGB
    imageGREY = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)  # RGB to GRAY
    bluredImage = cv2.medianBlur(imageGREY, 5)  # blur for circles
    return bluredImage


def circleOnImage(dataDict):
    falseCircles = 0;
    for data in dataDict:
        path = data["path"] + "/images/" + data["fileName"]
        circles4 = None
        circles5 = None
        image = cv2.imread(path)
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
                if (i[0] < 1000 and i[1] < 1000 and i[2] < 1000):
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
    return dataDict


def main():
    dataTrain = makingDictionaryForLearning()
    dataTrain = addingPartsToTrainData(dataTrain)
    learningBOW(dataTrain)
    dataTrain = extract(dataTrain)  # dictionary with added descriptor parameters
    afterTrain = train(dataTrain)

    x = input()
    if x == "detect":
        dataTest = makingDictionaryForTest()
        dataTest = circleOnImage(dataTest)
        dataTest = extract(dataTest)
        dataTest = predictImage(afterTrain, dataTest)
        detectInformation(dataTest)
        # printingChosenparts(dataTest)
    elif x == "classify":
        dataClassify = classifyInput()
        dataClassify = extract(dataClassify)
        dataClassify = predictImage(afterTrain, dataClassify)
        classifyReturn(dataClassify)
        # printingChosenparts(dataClassify)


if __name__ == '__main__':
    main()
