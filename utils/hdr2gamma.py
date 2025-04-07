import cv2
import skimage
import ezexr
import os
import numpy as np

dir = 'data/self/single'
target_dir = 'data/self/single_boost'

gamma = 2.2

files = os.listdir(dir)
files = ['C34.hdr']

for file in files:
    print(file)
    if not file.endswith('.hdr'):
        continue
    os.makedirs(os.path.join(target_dir, file.replace('.hdr', '')), exist_ok=True)
    hdr = cv2.imread(os.path.join(dir, file), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    for num in [0, -1, -2, -3]: #evs[-1] is -5
        lumi = np.clip(((2 ** num) * hdr) ** (1/gamma), 0, 1)
        skimage.io.imsave(os.path.join(target_dir, file.replace('.hdr', ''), f"{int(num)}.png"), skimage.img_as_ubyte(lumi))