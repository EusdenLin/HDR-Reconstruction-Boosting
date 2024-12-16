from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.interpolate import interp1d
# def loadExposureSeq(path):
#     images = []
#     times = [1, 2, 4, 8]
#     with open(os.path.join(path, 'list.txt')) as f:
#         content = f.readlines()
#     print(content)
#     for line in content:
#         tokens = line.split()
#         images.append(cv.imread(os.path.join(path, tokens[0])))
#         times.append(1 / float(tokens[1]))
 
#     return images, np.asarray(times, dtype=np.float32)
 
# parser = argparse.ArgumentParser(description='Code for High Dynamic Range Imaging tutorial.')
# parser.add_argument('--input', type=str, help='Path to the directory that contains images and exposure times.')
# args = parser.parse_args()
 
# if not args.input:
#     parser.print_help()
#     exit(0)
 
 
# images, times = loadExposureSeq(args.input)

path = './CEVR/t60'

images = []
times = np.array([1, 1/2, 1/4, 1/8], dtype=np.float32)  

for i in range(4):
    images.append(cv.imread(os.path.join(path, '{}.png'.format(-i))))
 
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)

merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)

print(hdr.max(), hdr.min())
 
# tonemap = cv.createTonemap(2.2)
# ldr = tonemap.process(hdr)
 
# merge_mertens = cv.createMergeMertens()
# fusion = merge_mertens.process(images)
 
# cv.imwrite('fusion.png', fusion * 255)
# cv.imwrite('ldr.png', ldr * 255)
cv.imwrite('hdr.hdr', hdr)

response = response.reshape((256, 3))

ldr_indices = np.arange(256)

hdr = hdr/10/2
inverse_crf = response / 10

tone_mapped_image = np.zeros_like(hdr)  # Prepare output image
for c in range(3):  # Loop over RGB channels
    # Create an interpolation function for the channel
    interp_func = interp1d(inverse_crf[:, c], ldr_indices, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # Apply the interpolation to map HDR to LDR
    tone_mapped_image[..., c] = interp_func(hdr[..., c])

ldr = np.clip(tone_mapped_image, 0, 255).astype(np.uint8)
print(ldr.max(), ldr.min())
print(ldr.shape)
cv.imwrite('ldr.png', ldr)

# print(tone_mapped_image.max(), tone_mapped_image.min()) 

 

plt.figure
plt.plot(response[:, 0], 'r')
plt.plot(response[:, 1], 'g')
plt.plot(response[:, 2], 'b')
plt.grid(False)
plt.savefig('response.png')

# 1. store the CRF
# 2. use 