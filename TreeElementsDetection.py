import cv2
import xml.etree.ElementTree as ET
import os

tree = ET.parse(f'D:\programy_SI\PROJEKT_SI\/annotations\/road316.xml')
root = tree.getroot()
# print(root)
# for neighbor in root.iter('xmin'):
#     print(neighbor.attrib)
# for child in root:
#     print(child.tag, child.attrib)

print(root.find('filename').text)
print('czy tak:', root.find('size').find('width').text)
for object in root.findall('object'):
    bndbox = object.find('bndbox')
    print(bndbox[0].text)               #xmin
    print(bndbox[1].text)               #ymin
    print(bndbox[2].text)               #xmax
    print(bndbox[3].text)               #ymax
    name = object.find('name')
    print(name.text)
for object in root.findall('size'):
    print(object.find('width').text)
    print(object.find('height').text)
