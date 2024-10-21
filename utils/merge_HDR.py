import cv2
import numpy as np
import glob

def apply_gamma_correction(images, gamma=2.2):
    corrected_images = []
    inv_gamma = 1.0 / gamma
    for img in images:
        corrected_img = np.power(img / 255.0, inv_gamma)
        corrected_images.append(corrected_img)
    return corrected_images

def merge_hdr(image_paths, exposure_times):
    gamma = 2.2

    images = [cv2.imread(img_path).astype(np.uint8) for img_path in image_paths]

    for i in range(len(images)):
        images[i] = images[i] / 255.0  # Normalize to [0, 1]
        images[i] = np.power(images[i], 2.2)  # Inverse gamma correction
        images[i] = (images[i] * 255.0).astype(np.uint8)  # Back to [0, 255]

    merge_debevec = cv2.createMergeDebevec()

    hdr = merge_debevec.process(images, times=np.array(exposure_times, dtype=np.float32))

    tonemap = cv2.createTonemap(gamma=1)  # Adjust gamma for tonemapping
    ldr_image_tonemapped = tonemap.process(hdr)

    ldr_image_gamma_corrected = np.power(ldr_image_tonemapped, 1/gamma)
    ldr = np.clip(ldr_image_gamma_corrected * 255, 0, 255).astype(np.uint8)

    return hdr, ldr

def save_image(image, path):
    cv2.imwrite(path, image)

def main():
    # image_paths = sorted(glob.glob("./0926_gamma_inverse/t60/1st_results/*.png"))
    image_paths = sorted(glob.glob("./0926_gamma_inverse/test/*.png"))
    print(image_paths)

    exposure_times = [1/16, 1/32, 1/64, 1/8, 1/4, 1/2, 1] 

    # 合併 HDR 圖像
    hdr, ldr = merge_hdr(image_paths, exposure_times)

    # 保存 HDR 和 LDR 圖像
    save_image(ldr, "output_ldr.png")
    cv2.imwrite("output_hdr.hdr", hdr)

if __name__ == "__main__":
    main()