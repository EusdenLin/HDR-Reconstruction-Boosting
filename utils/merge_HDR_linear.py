import cv2
import glob
import numpy as np

# Function to linearize the LDR images
def linearize(image, gamma=2.2):
    return np.power(image / 255.0, gamma)

# Function to merge LDR stack into an HDR image
def merge_ldr_stack(image_paths, exposure_times, gamma=2.2):
    # Load LDR images
    ldr_images = [cv2.imread(img_path).astype(np.uint8) for img_path in image_paths]

    for i in range(len(ldr_images)):
        ldr_images[i] = ldr_images[i] / 255.0  # Normalize to [0, 1]
        ldr_images[i] = np.power(ldr_images[i], 2.2)  # Inverse gamma correction
        ldr_images[i] = (ldr_images[i] * 255.0).astype(np.uint8)  # Back to [0, 255]


    # Initialize an empty HDR image
    hdr_image = np.zeros_like(ldr_images[0], dtype=np.float32)
    weight_sum = np.zeros_like(ldr_images[0], dtype=np.float32)

    # Process each LDR image
    for i, ldr in enumerate(ldr_images):
        # Linearize the LDR image by removing gamma correction
        linear_image = linearize(ldr, gamma)
        
        # Compute the weight function (simple mid-tone emphasis)
        weight = linear_image * (1 - linear_image)
        
        # Scale the linearized image by its exposure time
        weighted_image = linear_image * exposure_times[i]

        # Accumulate weighted image into the HDR result
        hdr_image += weighted_image * weight
        weight_sum += weight

    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1
    
    # Final HDR image
    hdr_image /= weight_sum

    return hdr_image

def save_image(image, path):
    cv2.imwrite(path, image)

def main():
    # image_paths = sorted(glob.glob("./0926_gamma_inverse/t60/1st_results/*.png"))
    image_paths = sorted(glob.glob("./0926_gamma_inverse/test/*.png"))
    print(image_paths)

    exposure_times = [1/16, 1/32, 1/64, 1/8, 1/4, 1/2, 1] 

    # 合併 HDR 圖像
    hdr = merge_ldr_stack(image_paths, exposure_times)
    cv2.imwrite("output_hdr.hdr", hdr)

    hdr = hdr/np.max(hdr)
    ldr_image = hdr 

    gamma = 2.2
    
    # Scale back to [0, 255] and clip values
    # ldr_image_gamma_corrected = np.power(ldr_image, 1/gamma)
    ldr_image_tonemapped = np.clip(ldr_image * 255, 0, 255).astype('uint8')


    # 保存 HDR 和 LDR 圖像
    save_image(ldr_image_tonemapped, "output_ldr.png")
    print("HDR image saved as 'output_hdr.hdr'")

if __name__ == "__main__":
    main()
