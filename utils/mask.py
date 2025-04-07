import cv2
import os
import numpy as np
from diffusers.utils import load_image

def create_mask(image, threshold=240):
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    image = image.astype(np.float16)
    mask = np.zeros_like(image, dtype=np.float16)

    mask[image > threshold] = np.clip(255 * (image[image > threshold] - threshold)/(255 - threshold) , 0, 255)
    mask = mask.astype(np.uint8)
    mask = mask.max(axis=2)
    return mask

def save_image(image, path):
    cv2.imwrite(path, image)

def main():
    img_size = (1024, 1024)
    test_case = os.listdir("./data/special/gamma")
    for case in test_case:
        image = load_image(f"./data/special/gamma/{case}/0.png").resize(img_size)
        mask = create_mask(image)
        save_image(mask, f"./data/special/gamma/{case}/mask.png")

if __name__ == "__main__":
    main()