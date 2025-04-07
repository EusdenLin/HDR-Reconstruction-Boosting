import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_dynamic_range(image, is_hdr=False):
    """
    Compute the dynamic range of an image.
    :param image: Input image (HDR or LDR)
    :param is_hdr: Boolean flag for HDR images
    :return: Dynamic range in ratio and stops
    """
    if not is_hdr:
        # Convert LDR to linear space (assuming sRGB gamma correction)
        image = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    
    # Avoid zero values to prevent division issues
    min_val = np.min(image[image > 0])  # Smallest nonzero value
    max_val = np.max(image)
    
    dynamic_range_ratio = max_val / min_val
    dynamic_range_stops = np.log2(dynamic_range_ratio)
    
    return dynamic_range_ratio, dynamic_range_stops

def load_image(path, is_hdr=False):
    """Load an image in HDR or LDR format."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    if not is_hdr:
        image = image.astype(np.float32) / 255.0  # Normalize LDR to [0,1]
    
    return image

def main(ldr_path, hdr_path):
    # Load images
    ldr_image = load_image(ldr_path, is_hdr=False)
    hdr_image = load_image(hdr_path, is_hdr=True)
    
    # Compute dynamic ranges
    ldr_ratio, ldr_stops = compute_dynamic_range(ldr_image, is_hdr=False)
    hdr_ratio, hdr_stops = compute_dynamic_range(hdr_image, is_hdr=True)
    
    # Print results
    print(f"LDR Dynamic Range: {ldr_ratio:.2f} ({ldr_stops:.2f} stops)")
    print(f"HDR Dynamic Range: {hdr_ratio:.2f} ({hdr_stops:.2f} stops)")
    
    # Histogram visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(ldr_image.ravel(), bins=256, range=[0, 1], alpha=0.7, color='blue')
    plt.title("LDR Histogram (Gamma-corrected)")
    
    plt.subplot(1, 2, 2)
    plt.hist(hdr_image.ravel(), bins=256, alpha=0.7, color='red', log=True)
    plt.title("HDR Histogram (Linear)")
    
    plt.savefig('dynamic_range_histograms.png', dpi=300)
    
if __name__ == "__main__":
    ldr_path = "results/VDS/RH_TMO/CEVR/t60/inpaint.png"  # Replace with your LDR image path
    hdr_path = "results/VDS/RH_TMO/CEVR/t60/inpaint.hdr"  # Replace with your HDR image path
    main(ldr_path, hdr_path)