import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
# from sklearn.metrics import pairwise_distances
import cv2

def preprocess_image(image_paths):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (299, 299))
        image = image.astype('float32') / 255.0
        images.append(image)
    images = np.array(images)
    return images

def calculate_fid_score(generated_image_path, real_image_path):
    inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                              weights="imagenet", 
                              pooling='avg')
    
    real_image = preprocess_image(real_image_path)
    generated_image = preprocess_image(generated_image_path)
    real_features = inception_model.predict(real_image)
    generated_features = inception_model.predict(generated_image)
    
    print(f"Real features shape: {real_features.shape}")
    print(f"Generated features shape: {generated_features.shape}")

    # Calculate mean and covariance for real samples
    mu_real = np.mean(real_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)

    # Calculate mean and covariance for generated samples
    mu_generated = np.mean(generated_features, axis=0)
    cov_generated = np.cov(generated_features, rowvar=False)

    # Calculate squared root of product of covariances
    cov_sqrt = sqrtm(cov_real.dot(cov_generated))
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Calculate FID score
    fid_score = np.sum((mu_real - mu_generated) ** 2) + np.trace(cov_real + cov_generated - 2 * cov_sqrt)

    return fid_score

# Example usage
real_image_paths = ["../data/strength_test/t60/ref/-1.png", "../data/strength_test/t60/ref/-2.png", "../data/strength_test/t60/ref/-3.png"]  # 替換為實際的圖像路徑
generated_image_paths = ["../data/strength_test/t60/val/EV-1.png", "../data/strength_test/t60/val/EV-2.png", "../data/strength_test/t60/val/EV-3.png"]

fid_score = calculate_fid_score(generated_image_paths, real_image_paths)
print("FID score:", fid_score)