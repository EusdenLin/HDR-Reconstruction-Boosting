import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.io

img_size = (1024, 1024)
method = "CEVR"
data_folder = "data_self"
output_folder = "./"
test_case = 'C24'
ldr_indices = np.arange(256)
# Merge to HDR
    
def readLDR(images_path):
    images = []
    norm_value = 255.0
    n = len(images_path)
    for i in range(n):
        print('Reading image: {}'.format(images_path[i]))
        img = cv2.resize(cv2.cvtColor(cv2.imread(images_path[i], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), img_size)
        img = img.astype(np.float32)
        img = img / norm_value
        images.append(img.copy())

    return images, norm_value

def weight_fun(img, weight_type, bMeanWeight=0, bounds=[0, 1]):
    weight = np.zeros_like(img)

    if weight_type == 'DEB97':
        Z_min = bounds[0]
        Z_max = bounds[1]
        tr = (Z_min + Z_max) / 2
        delta = Z_max - Z_min
        indx1 = img <= tr
        indx2 = img > tr
        weight[indx1] = img[indx1] - Z_min
        weight[indx2] = Z_max - img[indx2]

        if delta > 0:
            weight = weight / delta


    elif weight_type == 'saturation':
        weight = img

    elif weight_type == 'noise':
        weight = 1 - img

    else:
        raise ValueError('Unknown weight type: {}'.format(weight_type))
    
    np.clip(weight, 0, 1)

    return weight

def removeCRF(img, response):
    img = img.astype(np.float32)
    col = img.shape[2]
    total_value = response.shape[0]

    delta = 1.0 / (total_value)
    x = np.arange(0, 1, delta)
    imgOut = np.zeros_like(img)

    for i in range(col):   
        interp_func = interp1d(x, response[:, i], kind='nearest', bounds_error=False, fill_value='extrapolate')

        # Apply interpolation to each pixel in channel i
        imgOut[..., i] = interp_func(img[..., i])  # Vectorized computation

    return imgOut


# path = './data/Deep_/t60'
# path = f'./data/Deep_Recursive_HDRI/t28'
inverse_crf = scipy.io.loadmat(f'./{data_folder}/{method}/{test_case}/response.mat')['lin_fun'].reshape((256, 3)).astype(np.float32)

images_path = []
times = np.array([8, 4, 2, 1], dtype=np.float32)  

for i in range(0, 4):
    images_path.append(f'./results_self/gamma/{test_case}/inpaint/{str(-i)}.png')

images, norm_value = readLDR(images_path)

delta_value = 1.0 / 65535.0
scale = 1.0

i_sat = np.argmin(times)
i_noise = np.argmax(times)

threshold = 0.9

imgOut = np.zeros_like(images[0].astype(np.float32))
total_weight = np.zeros_like(images[0].astype(np.float32))

for i in range(len(images)):
    weight_type = 'DEB97'
    tempstack = np.clip(images[i].astype(np.float32) / scale, 0, 1)
    if(i == i_sat):
        if(np.sum(tempstack > threshold) > 0):
            weight_type = 'saturation'

    if(i == i_noise):
        if(np.sum(tempstack < threshold) > 0):
            weight_type = 'noise'

    weight = weight_fun(tempstack, weight_type, 0, [0, 1])
    weight[tempstack < delta_value] = 0 

    # linearize the image
    tempstack = removeCRF(tempstack, inverse_crf)

    dt_i = times[i]

    # merge type: log
    imgOut = imgOut + weight * (np.log(tempstack + delta_value) - np.log(dt_i))
    total_weight = total_weight + weight

imgOut = imgOut / total_weight
imgOut = np.exp(imgOut)
saturation = 1e-4

# saturation check
if np.sum(total_weight <= saturation) > 0:
    i_med = np.round(times.shape[0] / 2).astype(np.int8)
    med = np.clip(images[i_med] / scale, 0, 1)
    tempstack = np.clip(images[i_sat].astype(np.float32) / scale, 0, 1)

    img_sat = removeCRF(tempstack, inverse_crf)
    img_sat = img_sat / times[i_sat]

    mask = np.ones_like(total_weight).astype(np.int8)
    mask[total_weight > saturation] = 0
    mask[med > 0.5] = 0
    mask = np.expand_dims(mask, axis=2)

    if np.max(mask) > 0.5:
        print('saturation')

        for i in range(3):
            io_i = imgOut[:, :, i]
            is_i = img_sat[:, :, i]
            io_i[mask] = is_i[mask]
            imgOut[:, :, i] = io_i
    
hdr = imgOut

for i in range(0, 3):
    hdr_scaled = hdr * pow(2, i)

    tone_mapped_image = np.zeros_like(hdr)  # Prepare output image
    for c in range(3):  # Loop over RGB channels
        # Create an interpolation function for the channel
        interp_func = interp1d(inverse_crf[:, c], ldr_indices, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Apply the interpolation to map HDR to LDR
        tone_mapped_image[..., c] = interp_func(hdr_scaled[..., c])

    ldr = np.clip(tone_mapped_image, 0, 255).astype(np.uint8)
    ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./{str(i-3)}.png', ldr)