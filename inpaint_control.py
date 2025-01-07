import numpy as np
import os
import cv2
import subprocess
import shutil
import random
from PIL import Image
from diffusers import AutoPipelineForInpainting, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from relighting.pipeline_xlinpaint import CustomStableDiffusionXLControlNetInpaintPipeline, CustomStableDiffusionControlNetInpaintPipeline
import cv2
from diffusers.utils import load_image
import torch
from transformers import pipeline as transformers_pipeline 

device = "cuda"

img_size = (1024, 1024)
# results_folder = "results"
# data_folder = "data"
# output_folder = "results_intermediate"
method = "gamma"
results_folder = "results_HDReye"
data_folder = "data_HDReye"
output_folder = "results_intermediate_HDReye"

# test_cases = ['t81', 't24', 't69', 't4', 't7', 't30', 't48', 't52', 't23', 't17', 't82', 't78', 't51', 't91', 't63', 't84', 't16', 't90', 't70', 't14', 't32', 't46', 't89', 't22', 't2', 't56', 't50', 't74', 't60', 't64', 't47', 't66', 't68', 't94', 't79', 't43']
test_cases = None
iterations = 4

strengths = [1, 0.95, 0.95, 0.95, 0.95]
compensation_scale = [0.0, 0.4, 0.4, 0.4, 0.5]

if test_cases is None:
  test_cases = os.listdir(f"./{data_folder}/{method}")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16"
).to(device)


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
extra_kwargs = {"vae": vae} if vae is not None else {}

case_count = 0

for test_case in test_cases:
  case_count += 1
  print(test_case, f', {case_count}/{len(test_cases)}')
  os.makedirs(f'./{output_folder}/{method}/{test_case}/', exist_ok=True)
  pipe = CustomStableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    **extra_kwargs,
  ).to("cuda")

  img_path = f"./{data_folder}/{method}/{test_case}/0.png"
  image = load_image(img_path).resize(img_size)

  mask_path = f"./{data_folder}/{method}/{test_case}/mask.png"
  
  prompt_path = f"./{data_folder}/{method}/{test_case}/caption_cog.txt"
  depth_path = f"./{data_folder}/{method}/{test_case}/depth.png"

  control_image = load_image(depth_path).resize(img_size)

  # prepare mask

  # prepare control image
  # depth_estimator = transformers_pipeline("depth-estimation", device=device)
  # control_image = depth_estimator(image)['depth']
  # control_image.save(depth_path)
  control_image = load_image(depth_path).resize(img_size)

  # prepare prompt
  with open(prompt_path, "r") as f:
    prompt = f.read()
  # prompt = "a photo of clear sky. High resuloion image with a lot of details and sharpness. 4K, Ultra Quality."
  negative_prompt = "ugly, dark, bad, terrible, awful, horrible, disgusting, gross, nasty, unattractive, unpleasant, repulsive, revolting, vile, foul, abhorrent, loathsome, hideous, unsightly, unlovely, unpleasing, unappealing, uninviting, unwelcome, unattractive, unprepossessing, uncomely, unbeautiful"

  for iteration in range(1, iterations+1):
    os.makedirs(f'./{output_folder}/{method}/{test_case}/{str(iteration)}_results/', exist_ok=True)
    generator = torch.Generator(device="cuda")
    seed = random.randint(0, 1000000)
    print(seed)

    for i in range(0, 4):
      generator = torch.Generator(device="cuda")
      generator.manual_seed(seed)
      exposure = str(-i)
      if iteration == 1:
        img_path = f"./{data_folder}/{method}/{test_case}/{exposure}.png"
      else:
        img_path = f"./{output_folder}/{method}/{test_case}/{str(iteration-1)}_tone_mapped_residual/{exposure}.png"
        mask_path = f"./{output_folder}/{method}/{test_case}/{str(iteration-1)}_tone_mapped_residual/mask_{str(-i)}.png"
        

      if i == 0:
        img_path = f"./{data_folder}/{method}/{test_case}/0.png"
        image = load_image(img_path).resize(img_size)
        image.save(f'./{output_folder}/{method}/{test_case}/{str(iteration)}_results/{exposure}.png')
        continue
      image = load_image(img_path).resize(img_size)
      mask_image = load_image(mask_path).resize(img_size)

      kwargs = {
            # "prompt_embeds": prompt_embeds,
            "prompt": prompt,
            'negative_prompt': negative_prompt,
            'num_inference_steps': 50,
            'generator': generator,
            'image': image,
            'mask_image': mask_image,
            'control_image': control_image,
            'strength': strengths[iteration],
            'inpaint_kwargs': {
                'strength': 0.1,
                'weight': 0.3,
                'method': 'normal',
                'filter': False,
                'another_seed' : iteration > 1,
            },
            'current_seed': seed,
            'controlnet_conditioning_scale': 0.5,
            'height': 1024,
            'width': 1024,
            'guidance_scale': 5.0,
        }

      with torch.no_grad():
        output_img = pipe(
          **kwargs
        ).images

      for i, img in enumerate(output_img):
        # clip and paste the image
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        mask = np.asarray(mask_image)
        mask = mask.astype(np.float64) / 255
        cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{str(iteration)}_results/{exposure}.png', img*mask + image*(1-mask))
        # img.save(f'./{output_folder}/{test_case}/{str(iteration)}_results/{exposure}.png')

    # Merge to HDR

    command = [
      "python3", "exposure2hdr.py",
      "--input_dir", f"{output_folder}/{method}/{test_case}/{str(iteration)}_results",
      "--output_dir", f"{output_folder}/{method}/{test_case}/",
      "--iteration", f"{str(iteration)}"
    ]
    subprocess.run(command)

     # Residual
    os.makedirs(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped_residual/', exist_ok=True)
    for i in range(-3, 0):
      # Load the image
      image = cv2.imread(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped/{str(i)}.png').astype(np.float32)
      baseline = cv2.imread(f'./{data_folder}/{method}/{test_case}/{str(i)}.png').astype(np.float32)
      mask = cv2.imread(f'./{data_folder}/{method}/{test_case}/mask.png', cv2.IMREAD_GRAYSCALE)

      baseline = cv2.resize(baseline, (1024, 1024))

      # Convert the image from BGR to YUV color space
      yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
      yuv_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2YUV)

      # Split the YUV channels
      Y1, U1, V1 = cv2.split(yuv_image)
      Y2, U2, V2 = cv2.split(yuv_baseline)

      residual = np.zeros(Y1.shape, dtype=np.float32)
      residual[Y1 < Y2] = Y2[Y1 < Y2] - Y1[Y1 < Y2]
      residual = cv2.bitwise_and(residual, residual, mask=mask)

      Y1_com = cv2.add(Y1, residual*compensation_scale[iteration])
      Y1_com = np.clip(Y1_com, 0, 255)

      cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped_residual/residual_{str(i)}.png', residual*10)

      # Merge the channels back together
      yuv_image = cv2.merge([Y1_com, U1, V1])

      # Convert the image back to BGR color space
      output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

      cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped_residual/{str(i)}.png', output_image)

      new_mask = np.clip((Y1 < Y2) * ((mask > 0).reshape(1024, 1024)) * 255, 0, 255).astype(np.uint8)    

      cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped_residual/mask_{str(i)}.png', new_mask)
    
  # Save to results
  os.makedirs(f'./{results_folder}/{method}/{test_case}/baseline', exist_ok=True)
  os.makedirs(f'./{results_folder}/{method}/{test_case}/inpaint', exist_ok=True)

  for i in range(1, 4):
    shutil.copy(f'./{output_folder}/{method}/{test_case}/{str(iterations)}_tone_mapped/{str(-i)}.png', f'./{results_folder}/{method}/{test_case}/inpaint/{str(-i)}.png')
    img = cv2.imread(f'./{data_folder}/{method}/{test_case}/{str(-i)}.png')
    img = cv2.resize(img, (1024, 1024))
    cv2.imwrite(f'./{results_folder}/{method}/{test_case}/baseline/{str(-i)}.png', img)
  
  img = cv2.imread(f'./{data_folder}/{method}/{test_case}/0.png')
  img = cv2.resize(img, (1024, 1024))
  cv2.imwrite(f'./{results_folder}/{method}/{test_case}/inpaint/-0.png', img)
  cv2.imwrite(f'./{results_folder}/{method}/{test_case}/baseline/-0.png', img)