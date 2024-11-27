import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from scipy import linalg
import os

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

def calculate_kid_score(real_image_paths, generated_image_paths):
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    real_images = torch.cat([preprocess_image(path) for path in real_image_paths])
    generated_images = torch.cat([preprocess_image(path) for path in generated_image_paths])

    with torch.no_grad():
        real_features = inception_model(real_images).detach().numpy()
        generated_features = inception_model(generated_images).detach().numpy()

    mu_real = np.mean(real_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    mu_generated = np.mean(generated_features, axis=0)
    cov_generated = np.cov(generated_features, rowvar=False)

    cov_mean = linalg.sqrtm(cov_real.dot(cov_generated))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    kid_score = np.sum((mu_real - mu_generated) ** 2) + np.trace(cov_real + cov_generated - 2 * cov_mean)
    return kid_score

target_folder = './1111_evaluation'
real_folder = './1125_VDS_fid'

real_image_paths = []
generated_image_paths = []

for path in os.listdir(target_folder):
    generated_image_paths.append(os.path.join(target_folder, path, '4_tone_mapped/tonemapped.png'))
    print(path)

print("++++++++++")

for path in os.listdir(real_folder):
    real_image_paths.append(os.path.join(real_folder, path, '1st_tone_mapped/tonemapped.png'))
    print(path)


kid_score = calculate_kid_score(real_image_paths, generated_image_paths)
print("KID score:", kid_score)

# KID score(gamma): 139.733
# KID score(ours): 180.086