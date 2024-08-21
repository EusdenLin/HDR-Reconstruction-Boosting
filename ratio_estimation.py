import cv2
import numpy as np

path = './data/strength_test/t60/val/'
mask = cv2.imread(path + 'mask.png')
mask = mask/255.0

for i in range(3, 0, -1):
    for j in range(i, -1, -1):
        img1 = cv2.imread(path + 'EV-' + str(i) + '.png')
        if j != 0:
            img2 = cv2.imread(path + 'EV-' + str(j) + '.png')
        else:
            img2 = cv2.imread(path + 'EV0.png')
        
        img1 = img1 * mask
        img2 = img2 * mask

        ratio = np.sum(img2) / np.sum(img1)
        print('EV-' + str(j) + ' / EV-' + str(i) + ' : ' + str(ratio))
        
        