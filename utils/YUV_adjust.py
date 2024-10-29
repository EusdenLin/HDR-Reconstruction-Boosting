import cv2
import os
import numpy as np

# cases = ["t68", "t78", "t91", "t95"]
cases = ["t60"]
input_folder = "1021_multi_mask"

iteration = '4'

def gamma_correction(image, gamma=2.2):
    return np.power(image / 255.0, gamma) * 255.0

img = np.array([0, 50, 100, 150, 200, 250])

generated = np.array([0, 45, 90, 135, 180, 225])

baseline = []
gen = []

img = gamma_correction(img)
print('img')
for i in range(0, 4):
    newimg = img * np.power(2.0, -i)
    newimg = np.clip(newimg, 0, 255)
    newimg = gamma_correction(newimg, 1/2.2)
    baseline.append(np.copy(newimg))

print('generated')
generated = gamma_correction(generated)
for i in range(0, 4):
    newimg = generated * np.power(2.0, -i)
    newimg = np.clip(newimg, 0, 255)
    newimg = gamma_correction(newimg, 1/2.2)
    gen.append(np.copy(newimg))

print('Comp1')
for i in range(0, 4):
    gen_img = gen[i]
    baseline_img = baseline[i]
    gen_img = np.clip(gen_img + 2*(baseline_img - gen_img), 0, 255)
    print(np.round(baseline_img, 2), '\n',np.round(gen_img, 2))

print('Comp2')
for i in range(0, 4):
    gen_img = gamma_correction(gen[i], 2.2)
    baseline_img = gamma_correction(baseline[i], 2.2)

    gen_img = np.clip(gen_img + 2*(baseline_img - gen_img), 0, 255)
    gen_img = gamma_correction(gen_img, 1/2.2)
    baseline_img = gamma_correction(baseline_img, 1/2.2)
    print(baseline_img.round(2), '\n', gen_img.round(2))

# print('generated2')
# for i in range(0, 4):
#     newimg = generated2 * np.power(2.0, -i)
#     print(gamma_correction(newimg, 1/2.2))


# for case in cases:
#     os.makedirs(f'./{input_folder}/{case}/YUV_adjust/', exist_ok=True)
#     for i in range(-3, 1):
        # Load the image
        # image = cv2.imread(f'./{input_folder}/{case}/{iteration}_tone_mapped/{str(i)}.png')
        # cevr = cv2.imread(f'./0926_gamma_inverse/{case}/{str(i)}.png')
        # mask = cv2.imread(f'./0926_gamma_inverse/{case}/mask.png', cv2.IMREAD_GRAYSCALE)



        # cevr = cv2.resize(cevr, (1024, 1024))

        # # Convert the image from BGR to YUV color space
        # yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # yuv_cevr = cv2.cvtColor(cevr, cv2.COLOR_BGR2YUV)

        # # Split the YUV channels
        # Y1, U1, V1 = cv2.split(yuv_image)
        # Y2, U2, V2 = cv2.split(yuv_cevr)

        # residual = np.zeros(Y1.shape, dtype=np.uint8)
        # residual[Y1 < Y2] = Y2[Y1 < Y2] - Y1[Y1 < Y2]
        # residual = cv2.bitwise_and(residual, residual, mask=mask)

        # Y1 = cv2.add(Y1, residual*5)
        # Y1 = np.clip(Y1, 0, 255)

        # # Merge the channels back together
        # yuv_image = cv2.merge([Y1, U1, V1])

        # # Convert the image back to BGR color space
        # output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        # # Save or display the modified image
        # cv2.imwrite(f'./{input_folder}/{case}/{iteration}_tone_mapped_residual/{str(i)}.png', output_image)
        
        # new_mask = np.clip((image < cevr) * (mask.reshape(1024, 1024)/255) * 255, 0, 255).astype(np.uint8)    
        # # new_mask = cv2.bitwise_and(new_mask, mask, mask=residual)

        # cv2.imwrite(f'./{input_folder}/{case}/{iteration}_tone_mapped_residual/mask_{str(i)}.png', new_mask)
