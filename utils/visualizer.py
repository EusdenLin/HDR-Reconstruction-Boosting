import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument('--image', type=str, default='results/VDS/RH_TMO/CEVR/t60/baseline.hdr', help='Path to the image file')
args.add_argument('--output', type=str, default='dynamic_range.png', help='Path to save the output image')
args.add_argument('--viz_row', type=int, default=200, help='Row to visualize the dynamic range')


def visualize_dynamic_range(image, viz_row=200, title="Image", cmap=None, output=None):
    r = image[viz_row, :, 0].astype(np.float32).reshape(-1)
    g = image[viz_row, :, 1].astype(np.float32).reshape(-1)
    b = image[viz_row, :, 2].astype(np.float32).reshape(-1)

    channels = {'R': r, 'G': g, 'B': b}
    dynamic_ranges = {}
    epsilon = 1e-6  # small constant to avoid division by zero
    for key, data in channels.items():
        # Only consider positive nonzero values
        valid_data = data[data > epsilon]
        if valid_data.size > 0:
            min_val = valid_data.min()
            max_val = valid_data.max()
            # Calculate dynamic range in dB
            dynamic_range_db = 20 * np.log10(max_val / min_val)
        else:
            dynamic_range_db = 0
        dynamic_ranges[key] = dynamic_range_db

    plt.figure(figsize=(12, 12))
    plt.xlim(0, image.shape[1])
    plt.ylim(0, 0.3)
    plt.title(title)
    plt.xlabel("Pixel index at row {}".format(viz_row))
    plt.ylabel("Pixel Intensity in HDR")
    plt.plot(r, color='red', label='R channel')
    plt.plot(g, color='green', label='G channel')
    plt.plot(b, color='blue', label='B channel')
    plt.legend()
    plt.grid()
    plt.savefig(output, dpi=300)

def visualize_viz_row(image, viz_row=200, title="Image", cmap=None):
    image[viz_row, :, :] = 0
    cv2.imwrite("viz_row.png", image)

if __name__ == "__main__":
    args = args.parse_args()
    image_path = args.image

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit(1)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Check if the image is HDR
    if image.dtype == np.uint16:
        print("Image is HDR")
    elif image.dtype == np.uint8:
        print("Image is LDR")
        image = image.astype(np.uint16) / 255.0
    else:
        print("Unknown image format")
    # Visualize the image
    visualize_dynamic_range(image, viz_row=args.viz_row, title="HDR visualization", cmap='gray', output=args.output)
    image = cv2.imread(image_path.replace('hdr', 'png'))

    visualize_viz_row(image, viz_row=args.viz_row)



[0.7071067811865475, 1.0, 0.35355339059327373, 0.5, 0.17677669529663687, 0.25, 0.125, 1.414213562373095, 2.82842712474619, 2.0, 5.65685424949238, 4.0, 8.0]


