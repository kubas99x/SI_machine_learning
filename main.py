import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
from sklearn import metrics
from os import walk

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

# "train" or "test"
def makingDictionaryForLearning(whatFolder):
    arrayWithDicionaries = []
    badDimensions = 0
    path = os.getcwd()
    upperPath = os.path.abspath(os.path.join(path, os.pardir))
    mypath = upperPath + "\/" + whatFolder + "\/annotations"
    filenames = next(walk(mypath), (None, None, []))[2]
    for object in filenames:
        tree = ET.parse(f'{mypath}\/{object}')
        root = tree.getroot()
        imageSize = root.find('size')
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
                    "fileName": root.find('filename').text,
                    "width": imageSize.find('width').text,
                    "height": imageSize.find('height').text,
                    "name": ob.find('name').text,
                    "bndbox": bndboxDict,
                    "path": upperPath + "\/" + whatFolder,
                    "status": status,           # 1 if speedsight, 2 if other
                }
                arrayWithDicionaries.append(imageDictionary)
    return arrayWithDicionaries

def learningBOW(imageDictionary):
    dictionarySize = 100
    bow = cv2.BOWKMeansTrainer(dictionarySize)
    sift = cv2.SIFT_create()
    for part in imageDictionary:
        image = cv2.imread(f'{part["path"]}\/images\/{part["fileName"]}')
        bndbox = part["bndbox"]
        sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
        try:
            grey = cv2.cvtColor(sightPart, cv2.COLOR_BGR2GRAY)
            kp = sift.detect(grey, None)
            kp, desc = sift.compute(grey, kp)
            # computing in rgb:
            # kp = sift.detect(sightPart, None)
            # kp, desc = sift.compute(sightPart, kp)
            if desc is not None:
                bow.add(desc)
        except:
            print("error in BOW")

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
        image = cv2.imread(f'{part["path"]}\/images\/{part["fileName"]}')
        bndbox = part["bndbox"]
        sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
        grey = cv2.cvtColor(sightPart, cv2.COLOR_BGR2GRAY)
        desc = bow.compute(grey, sift.detect(grey))  # input KeypointDescriptor, output imgDescriptor
        # computing in rgb
        # desc = bow.compute(sightPart, sift.detect(sightPart))  # input KeypointDescriptor, output imgDescriptor
        if desc is not None:
            part.update({'desc': desc})
        else:
            part.update({'desc': np.zeros((1, len(dictionary)))})
    return imageDictionary


def train(imageDictionary):
    clf = RandomForestClassifier(100)
    y_train = []
    x_train = np.empty((1, 100))
    for sample in imageDictionary:
        y_train.append(sample["status"])
        x_train = np.vstack((x_train, sample["desc"]))
    clf.fit(x_train[1:], y_train)
    return clf


def predictImage(rf, data):
    for sample in data:
        sample.update({"predictedStatus": rf.predict(sample['desc'])})


def accuracyCalculate(dataTest):
    trueStatus = []
    predictedStatus = []
    for data in dataTest:
        trueStatus.append(data["status"])
        predictedStatus.append(data["predictedStatus"])
    print("Accuracy:", metrics.accuracy_score(trueStatus, predictedStatus))


def printingTestImages(dataDict):
    for data in dataDict:
        image = cv2.imread(f'{data["path"]}\/images\/{data["fileName"]}')
        bndbox = data["bndbox"]
        sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
        cv2.imshow(f"status: {data['status']}, predicted: {data['predictedStatus']}", sightPart)
        print(f"status: {data['status']}, predicted: {data['predictedStatus']}")
        cv2.waitKey(0)


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
        path = data["path"] + "\/images\/" + data["fileName"]
        print("path: " ,path)
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
                xmin = i[0] - i[2] - 10
                xmax = i[0] + i[2] + 10
                ymin = i[1] - i[2] - 10
                ymax = i[1] + i[2] + 10
        except:
            falseCircles += 1

        # cv2.imshow(f"images{path_}", np.hstack([image]))
        # cv2.waitKey(0)
    print(falseCircles)


def main():
    # pathsWithSpeedSights, pathsWithOther = loadingImage()
    # summedPaths = pathsWithSpeedSights + pathsWithOther
    # circleOnImage(pathsWithSpeedSights)
    # print("making dictionary for Train Data")
    # dataTrain = makingDictionaryForLearning("train")
    # print("learning BOW")
    # learningBOW(dataTrain)
    # print("extract data Train")
    # dataTrain = extract(dataTrain)  # dictionary with added descriptor parameters
    # print("traning data")
    # afterTrain = train(dataTrain)
    # print("making dictionary for Test Data")
    # dataTest = makingDictionaryForLearning("test")
    # print("extract data Test")
    # dataTest = extract(dataTest)
    # print("predict Image")
    # predictImage(afterTrain, dataTest)
    # print("calculate Accuracy")
    # accuracyCalculate(dataTest)

    # printingTestImages(dataTest)

    dataTest = makingDictionaryForLearning("test")
    circleOnImage(dataTest)
if __name__ == '__main__':
    main()
