import cv2
import xml.etree.ElementTree as ET
import os
import random
import shutil
from os import walk

# "train" or "test"
def makingDictionaryForTest():
    arrayWithDicionaries = []
    path = os.getcwd()
    upperPath = os.path.abspath(os.path.join(path, os.pardir))
    mypath = upperPath + "\/test\/images"
    filenames = next(walk(mypath), (None, None, []))[2]
    for object in filenames:
        image = cv2.imread(object)
        height, width = image.shape
        imageDictionary = {
            "fileName": object,
            "width" : width,
            "height" : height,
            "path" : mypath
        }
        arrayWithDicionaries.append(imageDictionary)
    return arrayWithDicionaries

def sortingSpeedImages(imageNames):
    n = 0;
    for fileName in imageNames:
        img = cv2.imread(f'D:\programy_SI\PROJEKT_SI\images\{fileName}')
        if(n < 3):
            path = 'D:\programy_SI\PROJEKT_SI\speedLimitsTrain'
        else:
            path = 'D:\programy_SI\PROJEKT_SI\speedLimitsTest'
        cv2.imwrite(os.path.join(path, fileName), img)
        if n == 3:
            n = 0
        else:
            n+=1

def sortingOtherImages(imageNames):
    n = 0;
    for fileName in imageNames:
        img = cv2.imread(f'D:\programy_SI\PROJEKT_SI\images\{fileName}')
        if(n < 3):
            path = 'D:\programy_SI\PROJEKT_SI\otherTrain'
        else:
            path = 'D:\programy_SI\PROJEKT_SI\otherTest'
        cv2.imwrite(os.path.join(path, fileName), img)
        if n == 3:
            n = 0
        else:
            n+=1

def randomSorting(imageNames):
    path = os.getcwd()
    upperPath = os.path.abspath(os.path.join(path, os.pardir))
    testList = random.sample(imageNames,int(len(imageNames)/4))
    trainList = [item for item in imageNames if item not in testList]
    for object in testList:
        shutil.copy(f'{upperPath}\images\/road{object}.png', f'{upperPath}\/test\images\/road{object}.png')
        shutil.copy(f'{upperPath}\/annotations\/road{object}.xml', f'{upperPath}\/test\/annotations\/road{object}.xml')
    for object in trainList:
        shutil.copy(f'{upperPath}\images\/road{object}.png', f'{upperPath}\/train\images\/road{object}.png')
        shutil.copy(f'{upperPath}\/annotations\/road{object}.xml', f'{upperPath}\/train\/annotations\/road{object}.xml')
# i= 0
# speedLimitImageNames = [];
# otherImageNames = [];
# while os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml'):
#     tree = ET.parse(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml')
#     root = tree.getroot()
#     if(root[4][0].text == 'speedlimit'):
#         speedLimitImageNames.append(root[1].text)
#     else:
#         otherImageNames.append(root[1].text)
#     i+=1
#
# print(len(speedLimitImageNames))
# print(len(otherImageNames))

#sortingSpeedImages(speedLimitImageNames)
#sortingOtherImages(otherImageNames)
# i=0
# speedLimitImageNumbers = [];
# otherImageNumbers = [];
# while os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml'):
#     tree = ET.parse(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml')
#     root = tree.getroot()
#     nameArray = []
#     for object in root.findall('object'):
#         name = object.find('name').text
#         nameArray.append(name)
#     if "speedlimit" in nameArray:
#         speedLimitImageNumbers.append(i)
#     else:
#         otherImageNumbers.append(i)
#     i+=1


# randomSorting(speedLimitImageNumbers)
# randomSorting(otherImageNumbers)
path = os.getcwd()
upperPath = os.path.abspath(os.path.join(path, os.pardir))
mypath = upperPath + "\/train\/annotations"
filenames = next(walk(mypath), (None, None, []))[2]
print(filenames)

f = []
paths = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    paths.append(dirnames)
    break
print(paths)