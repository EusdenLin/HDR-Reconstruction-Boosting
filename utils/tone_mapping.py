import cv2
import os
import numpy as np
import ezexr

# Load the HDR image


# Create a Reinhard tone mapping object
tonemap_reinhard = cv2.createTonemapReinhard()

path_inpaint = './1111_evaluation'
path_inpaint_turbo = './1111_evaluation_turbo'
path_gamma = './1126_gamma'
path_cevr = './CEVR'
path_glowgan = './evaluation_glowgan_firework'

output_path = './evaluation_RH'

inpaint_paths = os.listdir(path_inpaint)

for path in inpaint_paths:
    print(path)
    # # glowgan
    # hdr_image = cv2.imread(f'{path_glowgan}/{path}.hdr', cv2.IMREAD_ANYDEPTH)
    # ldr_image = tonemap_reinhard.process(hdr_image)
    # ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
    # cv2.imwrite(f"{path_glowgan}/{path}_tm.png", ldr_image)

    # inpaint
    hdr_image = ezexr.imread(f'{path_inpaint}/{path}/4_tone_mapped/hdr.exr')
    ldr_image = tonemap_reinhard.process(hdr_image)
    ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_path}/{path}/inpaint.png", ldr_image)

    # # inpaint_turbo
    # hdr_image = ezexr.imread(f'{path_inpaint_turbo}/{path}/4_tone_mapped/hdr.exr')
    # ldr_image = tonemap_reinhard.process(hdr_image)
    # ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
    # cv2.imwrite(f"{output_path}/{path}/inpaint_turbo.png", ldr_image)

    # # gamma
    # hdr_image = ezexr.imread(f'{path_gamma}/{path}/1st_tone_mapped/hdr.exr')
    # ldr_image = tonemap_reinhard.process(hdr_image)
    # ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
    # cv2.imwrite(f"{output_path}/{path}/gamma.png", ldr_image)

    # # cevr
    # hdr_image = cv2.imread(f'{path_cevr}/{path}/cevr.hdr', cv2.IMREAD_ANYDEPTH)
    # ldr_image = tonemap_reinhard.process(hdr_image)
    # ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
    # cv2.imwrite(f"{output_path}/{path}/cevr.png", ldr_image)

