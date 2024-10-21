import cv2
import numpy as np

gamma = 2.2

# Load the HDR image (the one you've created)
hdr_image = cv2.imread('output_hdr.hdr', cv2.IMREAD_UNCHANGED)

# Normalize to [0, 1]
hdr_image = hdr_image / hdr_image.max()

# Define exposure settings for generating LDR images
# These should match the original exposure values you used
exposures = np.array([1, 1/2, 1/4, 1/8], dtype=np.float32)  # Adjust as needed

# Tonemap operator to visualize the HDR image
tonemap = cv2.createTonemap(gamma=1)  # Set gamma for tonemapping

# Generate LDR images based on exposures
ldr_images = []
for exposure in exposures:
    # Scale the HDR image based on exposure
    ldr_image = hdr_image * exposure
    
    # # Tonemap the scaled HDR image
    # ldr_image_tonemapped = tonemap.process(ldr_image)
    
    # Scale back to [0, 255] and clip values
    ldr_image_gamma_corrected = np.power(ldr_image, 1/gamma)
    ldr_image_tonemapped = np.clip(ldr_image_gamma_corrected * 255, 0, 255).astype('uint8')
    
    # Store the resulting LDR image
    ldr_images.append(ldr_image_tonemapped)

# Save the LDR images
for i, ldr_image in enumerate(ldr_images):
    cv2.imwrite(f'./0926_gamma_inverse/t60/tone_mapped/{-i}.png', ldr_image)

print("LDR images generated and saved.")
