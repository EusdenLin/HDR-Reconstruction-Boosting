from brisque import BRISQUE
import cv2
import os

def get_score(img_paths):
    avg_score = 0
    for path in img_paths:
        img = cv2.imread(path)
        obj = BRISQUE(url=False)
        score = obj.score(img)
        avg_score += score

    avg_score /= len(img_paths)
    return avg_score

target_folder = './1111_evaluation'

gamma_image_paths = []
generated_image_paths = []

for path in os.listdir(target_folder):
    generated_image_paths.append(os.path.join(target_folder, path, '4_tone_mapped/-1.png'))
    generated_image_paths.append(os.path.join(target_folder, path, '4_tone_mapped/-2.png'))
    generated_image_paths.append(os.path.join(target_folder, path, '4_tone_mapped/-3.png'))

for path in os.listdir(target_folder):
    gamma_image_paths.append(os.path.join(target_folder, path, '-3.png'))
    gamma_image_paths.append(os.path.join(target_folder, path, '-2.png'))
    gamma_image_paths.append(os.path.join(target_folder, path, '-1.png'))


gamma_score = get_score(gamma_image_paths)
generated_score = get_score(generated_image_paths)

print(f'Gamma score: {gamma_score}')
print(f'Generated score: {generated_score}')