import cv2
import skimage
import ezexr
import os
import numpy as np

dir = '/ssddisk/ytlin/data/HDR-Real/'
target_dir = '/ssddisk/ytlin/data/HDR-Real/single_boost'

gamma = 2.2

files = os.listdir(dir)

os.makedirs(target_dir, exist_ok=True)

for file in files:
    print(file)
    os.makedirs(os.path.join(target_dir, file), exist_ok=True)
    hdr = cv2.imread(os.path.join(dir, file, 'ours.hdr'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    for num in [0, -1, -2, -3]: #evs[-1] is -5
        lumi = np.clip(((2 ** num) * hdr) ** (1/gamma), 0, 1)
        skimage.io.imsave(os.path.join(target_dir, file, f"{int(num)}.png"), skimage.img_as_ubyte(lumi))