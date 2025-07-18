import cv2
import skimage
import ezexr
import os
import numpy as np

dir = '/home/ytlin/boosting_HDR/data/HDR-Real/glowgan'
target_dir = '/home/ytlin/boosting_HDR/data/HDR-Real/glowgan_boost'

gamma = 2.2

files = os.listdir(dir)

os.makedirs(target_dir, exist_ok=True)
print(len(files))
exit()
for file in files:
    if file.endswith('.png'):
        continue
    print(file)
    os.makedirs(os.path.join(target_dir, file.replace('.hdr', '')), exist_ok=True)
    hdr = cv2.imread(os.path.join(dir, file), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    for num in [0, -1, -2, -3]: #evs[-1] is -5
        lumi = np.clip(((2 ** num) * hdr) ** (1/gamma), 0, 1)
        skimage.io.imsave(os.path.join(target_dir, file.replace('.hdr', ''), f"{int(num)}.png"), skimage.img_as_ubyte(lumi))