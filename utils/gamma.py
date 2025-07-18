import cv2
import os
import numpy as np

def adjust_exposure(image, exposure_factor):
    # Normalize the image to [0, 1]
    image = image / 255.0

    # Apply gamma correction (inverse of gamma 2.2)
    linear_image = np.power(image, 2.2)

    # Adjust exposure by scaling the linear values
    adjusted_image = linear_image * exposure_factor

    # Clip values to [0, 1] to avoid overflow
    adjusted_image = np.clip(adjusted_image, 0, 1)

    # Convert back to gamma-corrected space
    gamma_corrected_image = np.power(adjusted_image, 1/2.2)

    # Scale back to [0, 255]
    gamma_corrected_image = (gamma_corrected_image * 255).astype(np.uint8)

    return gamma_corrected_image

# Load the image

# cases = ["t68", "t78", "t91", "t95"]
cases = os.listdir("./data/HDR-Real/glowgan")
os.makedirs("./data/HDR-Real/glowgan_boost", exist_ok=True)
for case in cases:
    print(case)
    if case.endswith(".png"):
        continue
    os.makedirs(f"./data/HDR-Real/glowgan_boost/{case.replace('.hdr', '')}", exist_ok=True)
    image = cv2.imread(f'./data/HDR-Real/GT_resize/{case}')    
    for i in range(0, 4):
        exposure = str(-i)
        img_path = f"./data/HDR-Real/glowgan/{case.replace('.hdr', '')}/{exposure}.png"

        # Set the desired exposure factor (e.g., 1.5 for +1.5 stops)
        exposure_factor = pow(2, -i)
        # Adjust the exposure
        adjusted_image = adjust_exposure(image, exposure_factor)

        # Save or display the adjusted image
        cv2.imwrite(img_path, cv2.resize(adjusted_image, (1024, 1024)))

        exposure = str(i)
        img_path = f"./data/HDR-Real/glowgan/{case.replace('.png', '')}/{exposure}.png"

        # Set the desired exposure factor (e.g., 1.5 for +1.5 stops)
        exposure_factor = pow(2, i)
        # Adjust the exposure
        adjusted_image = adjust_exposure(image, exposure_factor)

        # Save or display the adjusted image
        cv2.imwrite(img_path, cv2.resize(adjusted_image, (1024, 1024)))
