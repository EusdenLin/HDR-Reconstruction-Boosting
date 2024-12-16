from brisque import BRISQUE
import cv2
import os

def get_score(img_paths):
    avg_score = 0
    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (1024, 1024))
        obj = BRISQUE(url=False)
        score = obj.score(img)
        avg_score += score

    avg_score /= len(img_paths)
    return avg_score

# target_folder = './evaluation_RH'
# target_folder = './evaluation_glowgan'
target_folder = './hdr_images'

gamma_paths = []
inpaint_paths = []
inpaint_turbo_paths = []
glow_paths = []

for path in os.listdir(target_folder):
    # for i in range(1, 4):
    gamma_paths.append(f'{target_folder}/{path}/gamma.png')
    inpaint_paths.append(f'{target_folder}/{path}/inpaint.png')
    inpaint_turbo_paths.append(f'{target_folder}/{path}/inpaint_turbo.png')
    glow_paths.append(f'{target_folder}/{path}/glowgan.png')


gamma_score = get_score(gamma_paths)
inpaint_score = get_score(inpaint_paths)
inpaint_turbo_score = get_score(inpaint_turbo_paths)
glow_score = get_score(glow_paths)

print(f'Gamma score: {gamma_score}')
print(f'Generated score: {inpaint_score}')
print(f'Generated turbo score: {inpaint_turbo_score}')
print(f'GlowGAN score: {glow_score}')