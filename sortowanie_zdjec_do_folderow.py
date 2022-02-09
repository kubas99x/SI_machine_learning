import cv2
import xml.etree.ElementTree as ET
import os

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


i= 0
speedLimitImageNames = [];
otherImageNames = [];
while os.path.isfile(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml'):
    tree = ET.parse(f'D:\programy_SI\PROJEKT_SI\/annotations\/road{i}.xml')
    root = tree.getroot()
    if(root[4][0].text == 'speedlimit'):
        speedLimitImageNames.append(root[1].text)
    else:
        otherImageNames.append(root[1].text)
    i+=1

print(len(speedLimitImageNames))
print(len(otherImageNames))

#sortingSpeedImages(speedLimitImageNames)
#sortingOtherImages(otherImageNames)