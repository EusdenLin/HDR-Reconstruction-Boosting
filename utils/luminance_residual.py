import cv2
import numpy as np

# Load the image
# image = cv2.imread('./results_test/1st_bw/-3.png')
image = cv2.imread('./0916_multi_iter_lumi_com/3rd_com/results.png')
cevr = cv2.imread('./EV-3_CEVR.png')
mask = cv2.imread('./mask_1024.png', cv2.IMREAD_GRAYSCALE)

# Convert the image from BGR to YUV color space
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
yuv_cevr = cv2.cvtColor(cevr, cv2.COLOR_BGR2YUV)

# Split the YUV channels
Y1, U1, V1 = cv2.split(yuv_image)
Y2, U2, V2 = cv2.split(yuv_cevr)

residual = np.zeros(Y1.shape, dtype=np.uint8)
residual[Y1 < Y2] = Y2[Y1 < Y2] - Y1[Y1 < Y2]
residual = cv2.bitwise_and(residual, residual, mask=mask)

residual = residual * 3


# save 
cv2.imwrite('./0916_multi_iter_lumi_com/3rd_com/diff_lumi_emp.png', residual)