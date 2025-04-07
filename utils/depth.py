from PIL import Image
import depth_pro
import os
import numpy as np

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(device="cuda")
model.eval()

# Run inference.

path = './data/special/gamma/'
cases = os.listdir(path)

for case in cases:
    print(f'Processing {case}...')
    image, _, f_px = depth_pro.load_rgb(f'{path}{case}/0.png')
    image = transform(image)
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    depth = depth.squeeze().cpu().numpy()
    
    depth_log = np.log1p(depth) 
    depth_min = depth_log.min()
    depth_max = depth_log.max()
    depth_scaled = 255 * (depth_log - depth_min) / (depth_max - depth_min)
    depth_scaled = np.clip(depth_scaled, 0, 255).astype(np.uint8)

    depth_scaled = 255 - depth_scaled

    depth_image = Image.fromarray(depth_scaled)
    depth_image.save(f'{path}{case}/depth.png')
