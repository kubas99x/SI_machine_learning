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
#train or test
def makingDictionaryForLearning(whatFolder):
    arrayWithDicionaries = []
    path = os.getcwd()
    upperPath = os.path.abspath(os.path.join(path, os.pardir))
    mypath = upperPath + "\/" + whatFolder + "\/annotations"
    filenames = next(walk(mypath), (None, None, []))[2]
    for object in filenames:
        tree = ET.parse(f'{mypath}\/{object}')
        root = tree.getroot()
        imageSize = root.find('size')
        partDictionaryArray = []
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
                continue
            else:
                if ob.find('name').text == "speedlimit":
                    status = 1
                else:
                    status = 2
                partDicionary = {
                    "name" : ob.find('name').text,
                    "bndboxDict": bndboxDict,
                    "status" : status
                }
                partDictionaryArray.append(partDicionary)
        if not partDictionaryArray: #jezeli wszystkie wycinki w obrazie byly za male, pomijamy to zdjecie
            continue
        else:
            imageDictionary = {
                "fileName": root.find('filename').text,
                "width": imageSize.find('width').text,
                "height": imageSize.find('height').text,
                "path": upperPath + "\/" + whatFolder,
                "partDictionaries" : partDictionaryArray
                }
            arrayWithDicionaries.append(imageDictionary)
    return arrayWithDicionaries

def learningBOW(imageDictionary):
    dictionarySize = 100
    bow = cv2.BOWKMeansTrainer(dictionarySize)
    sift = cv2.SIFT_create()
    for part in imageDictionary:
        image = cv2.imread(f'{part["path"]}\/images\/{part["fileName"]}')
        for object in part["partDictionaries"]:
            bndbox = object["bndboxDict"]
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
    dictionary = np.load('dict.npy')
    bow.setVocabulary(dictionary)
    for part in imageDictionary:
        image = cv2.imread(f'{part["path"]}\/images\/{part["fileName"]}')
        if part["partDictionaries"]:                                                    #it could be when we are taking parts from circles
            for object in part["partDictionaries"]:
                bndbox = object["bndboxDict"]
                sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
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


def accuracyCalculate(dataTest):        #only with test images using xml
    trueStatus = []
    predictedStatus = []
    for data in dataTest:
        for object in data["partDictionaries"]:
            trueStatus.append(object["status"])
            predictedStatus.append(object["predictedStatus"])
    print("Accuracy:", metrics.accuracy_score(trueStatus, predictedStatus))


def printingTestImages(dataDict):
    for data in dataDict:
        image = cv2.imread(f'{data["path"]}\/images\/{data["fileName"]}')
        # bndbox = data["partDictionaries"]["bndboxDict"]
        # sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
        cv2.imshow(f'{data["path"]}\/images\/{data["fileName"]}', image)
        i = 0
        for tmp in data["partDictionaries"]:
            print(f'Znak: {i}',tmp["status"], tmp["predictedStatus"])
            i+=1
        cv2.waitKey(0)

def printingChoosenparts(dataTest):
    for data in dataTest:
        image = cv2.imread(f'{data["path"]}\/images\/{data["fileName"]}')
        #imageArray = None
        #imageArray = np.hstack([image])
        for part in data["partDictionaries"]:
            bndbox = part["bndboxDict"]
            sightPart = image[bndbox["ymin"]:bndbox["ymax"], bndbox["xmin"]:bndbox["xmax"]]
            #imageArray = np.hstack([imageArray, sightPart])
            cv2.imshow(f"{part['predictedStatus']}", np.hstack(image, sightPart))
            cv2.waitKey(0)
        # cv2.imshow(f"images", imageArray)
        # cv2.waitKey(0)
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
        data["partDictionaries"].clear()

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
                xmin = i[0] - i[2] - 5
                xmax = i[0] + i[2] + 5
                ymin = i[1] - i[2] - 5
                ymax = i[1] + i[2] + 5
                print(xmin, xmax, ymin, ymax)
                if(xmax - xmin > int(data["width"])/10 and  ymax - ymin > int(data["height"])/10):
                    print("IM HERE")
                    bndboxDict = {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax}
                    data["partDictionaries"].update({"bndboxDict":bndboxDict})
        except:
            falseCircles += 1
        data.update({"elementsOnImage" : len(data["partDictionaries"])}) #ilosc elementow wykrytych poprzez algorytm
        #print("length of dict: ",len(data['partDictionaries']))
        # cv2.imshow(f"images{path_}", np.hstack([image]))
        # cv2.waitKey(0)
    print(falseCircles)
    return dataDict

def main():
    # pathsWithSpeedSights, pathsWithOther = loadingImage()
    # summedPaths = pathsWithSpeedSights + pathsWithOther
    # circleOnImage(pathsWithSpeedSights)
    print("making dictionary for Train Data")
    dataTrain = makingDictionaryForLearning("train")
    print("learning BOW")
    learningBOW(dataTrain)
    print("extract data Train")
    dataTrain = extract(dataTrain)  # dictionary with added descriptor parameters
    print("traning data")
    afterTrain = train(dataTrain)
    # print("making dictionary for Test Data")
    # dataTest = makingDictionaryForLearning("test")
    # print("extract data Test")
    # dataTest = extract(dataTest)
    # print("predict Image")
    # predictImage(afterTrain, dataTest)
    # print("calculate Accuracy")
    # accuracyCalculate(dataTest)


    #printingTestImages(dataTest)

    print("making dictionary for Test Data")
    dataTest = makingDictionaryForLearning("test")
    print("searching for interesting places on image")
    dataTest = circleOnImage(dataTest)
    print("extract data Test")
    dataTest = extract(dataTest)
    print("predict Image")
    predictImage(afterTrain, dataTest)
    print("printing choosen pants")
    printingChoosenparts(dataTest)

    # dataTest = makingDictionaryForLearning("test")
    # circleOnImage(dataTest)


if __name__ == '__main__':
    main()
