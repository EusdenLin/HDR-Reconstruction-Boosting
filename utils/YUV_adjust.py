import cv2
import os
import numpy as np

cases = ["t68", "t78", "t91", "t95"]

iteration = '2'

for case in cases:
    os.makedirs(f'./0926_gamma_inverse/{case}/{iteration}_tone_mapped_residual/', exist_ok=True)
    for i in range(-3, 1):
        # Load the image
        image = cv2.imread(f'./0926_gamma_inverse/{case}/{iteration}_tone_mapped/{str(i)}.png')
        cevr = cv2.imread(f'./0926_gamma_inverse/{case}/{str(i)}.png')
        mask = cv2.imread(f'./0926_gamma_inverse/{case}/mask.png', cv2.IMREAD_GRAYSCALE)

        cevr = cv2.resize(cevr, (1024, 1024))

        # Convert the image from BGR to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_cevr = cv2.cvtColor(cevr, cv2.COLOR_BGR2YUV)

        # Split the YUV channels
        Y1, U1, V1 = cv2.split(yuv_image)
        Y2, U2, V2 = cv2.split(yuv_cevr)

        residual = np.zeros(Y1.shape, dtype=np.uint8)
        residual[Y1 < Y2] = Y2[Y1 < Y2] - Y1[Y1 < Y2]
        residual = cv2.bitwise_and(residual, residual, mask=mask)

        Y1 = cv2.add(Y1, residual*3)
        Y1 = np.clip(Y1, 0, 255)

        # Merge the channels back together
        yuv_image = cv2.merge([Y1, U1, V1])

        # Convert the image back to BGR color space
        output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        # Save or display the modified image
        cv2.imwrite(f'./0926_gamma_inverse/{case}/{iteration}_tone_mapped_residual/{str(i)}.png', output_image)
