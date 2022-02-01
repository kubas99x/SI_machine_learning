import numpy as np
import cv2



# define the list of boundaries
# B G R
# boundaries = [
#     ([17, 15, 100], [50, 56, 200]),
#     ([86, 31, 4], [220, 88, 50]),
#     ([25, 146, 190], [62, 174, 250]),
#     ([103, 86, 65], [145, 133, 128])
# ]
# loop over the boundaries
# for (lower, upper) in boundaries:
#     # create NumPy arrays from the boundaries
#     lower = np.array(lower, dtype="uint8")
#     upper = np.array(upper, dtype="uint8")
#     # find the colors within the specified boundaries and apply the mask
#     mask = cv2.inRange(image, lower, upper)
#     output = cv2.bitwise_and(image, image, mask=mask)
#     # show the images
#     cv2.imshow("images", np.hstack([image, output]))
#     cv2.waitKey(0)

# load the image
path=[]
for i in range(5):
    path.append(f"D:\programy_SI\PROJEKT_SI\znak_{i}_test.png")

#image = cv2.imread("D:\programy_SI\PROJEKT_SI\znak_1_test.png")

for path_ in path:
    image = cv2.imread(path_)
    mask = cv2.inRange(image, np.array([17, 15, 100]), np.array([50, 56, 200]))
    output = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)