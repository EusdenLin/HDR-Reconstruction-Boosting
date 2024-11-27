import cv2
import os
import numpy as np

def load_image(path):
    return cv2.imread(path)

def create_mask(image, threshold=250):
    image = image.astype(np.float16)
    mask = np.zeros_like(image, dtype=np.float16)

    mask[image > threshold] = np.clip(255 * (image[image > threshold] - threshold)/(255 - threshold) , 0, 255)
    mask = mask.astype(np.uint8)
    mask = mask.max(axis=2)
    return mask

def save_image(image, path):
    cv2.imwrite(path, image)

def main():
    # test_case = ["t60", "t68", "t82"]
    test_case = os.listdir("./1111_evaluation/")
    for case in test_case:
        image = load_image(f"./1111_evaluation/{case}/0.png")
        mask = create_mask(image)
        mask = cv2.resize(mask, (1024, 1024))
        save_image(mask, f"./1111_evaluation/{case}/mask.png")

if __name__ == "__main__":
    main()