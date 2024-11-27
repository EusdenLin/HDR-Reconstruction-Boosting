import numpy as np
import os
import cv2
import subprocess
import random
from PIL import Image
from diffusers import AutoPipelineForInpainting, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from relighting.pipeline_xlinpaint import CustomStableDiffusionXLControlNetInpaintPipeline, CustomStableDiffusionControlNetInpaintPipeline

from diffusers.utils import load_image
import torch
from transformers import pipeline as transformers_pipeline 

device = "cuda"

img_size = (1024, 1024)

# target_folder = "1111_evaluation"  
target_folder = "1119_diversity"  
# output_folder = "1111_evaluation"
output_folder = "1119_diversity"
# test_cases = None
test_cases = ["t60"]
iterations = 4

strengths = [1, 0.95, 0.95, 0.95, 0.95]

if test_cases is None:
  test_cases = os.listdir(f"./{target_folder}")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16"
).to(device)


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
extra_kwargs = {"vae": vae} if vae is not None else {}

for test_case in test_cases:
  print(test_case)
  os.makedirs(f'./{output_folder}/{test_case}/', exist_ok=True)
  pipe = CustomStableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    **extra_kwargs,
  ).to("cuda")

  img_path = f"./{target_folder}/{test_case}/0.png"
  # img_path = f"1021_multi_mask/t60/1_results/0.png"
  image = load_image(img_path).resize(img_size)

  mask_path = f"./{target_folder}/{test_case}/mask.png"
  
  prompt_path = f"./{target_folder}/{test_case}/caption_cog.txt"
  depth_path = f"./{target_folder}/{test_case}/depth.png"

  # prepare mask

  # prepare control image
  # depth_estimator = transformers_pipeline("depth-estimation", device=device)
  # control_image = depth_estimator(image)['depth']
  # control_image.save(depth_path)
  control_image = load_image(depth_path).resize(img_size)

  # prepare prompt
  with open(prompt_path, "r") as f:
    prompt = f.read()
  prompt = "a photo of clear sky with a few white clouds scattered around. The background is very very bright"

  for iteration in range(1, iterations+1):
    os.makedirs(f'./{output_folder}/{test_case}/{str(iteration)}_results/', exist_ok=True)
    generator = torch.Generator(device="cuda")
    seed = random.randint(0, 1000000)

    for i in range(0, 4):
      generator = torch.Generator(device="cuda")
      generator.manual_seed(seed)
      exposure = str(-i)
      if iteration == 1:
        img_path = f"./{target_folder}/{test_case}/{exposure}.png"
      else:
        img_path = f"./{target_folder}/{test_case}/{str(iteration-1)}_tone_mapped_residual/{exposure}.png"
        # img_path = f"./1021_multi_mask/{test_case}/{str(iteration-1)}_tone_mapped_residual/{exposure}.png"
        mask_path = f"./{target_folder}/{test_case}/{str(iteration-1)}_tone_mapped_residual/mask_{str(-i)}.png"
        

      mask_image = load_image(mask_path).resize(img_size)
      image = load_image(img_path).resize(img_size)
      image_2 = load_image(img_path).resize(img_size)
      # image_2 = load_image(f"./{target_folder}/tone_mapped/{test_case}/{exposure}.png").resize((1024, 1024))

      kwargs = {
            # "prompt_embeds": prompt_embeds,
            "prompt": prompt,
            # 'negative_prompt': args.negative_prompt,
            'num_inference_steps': 50,
            'generator': generator,
            'image': image,
            'image_2': image_2, # for consistency
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
        # img = np.array(image) * np.array(mask_image) + img * (1 - np.array(mask_image))
        # cv2.imwrite(f'./{output_folder}/{test_case}/{str(iteration)}_results/{exposure}.png', img)
        img.save(f'./{output_folder}/{test_case}/{str(iteration)}_results/{exposure}.png')
        # img.save(f'./1016_resizetest/{test_case}/{exposure}.png')

    # Merge to HDR

    command = [
      "python3", "exposure2hdr.py",
      "--input_dir", f"{output_folder}/{test_case}/{str(iteration)}_results",
      "--output_dir", f"{output_folder}/{test_case}/",
      "--iteration", f"{str(iteration)}"
    ]
    subprocess.run(command)

    # Residual
    os.makedirs(f'./{output_folder}/{test_case}/{iteration}_tone_mapped_residual/', exist_ok=True)
    for i in range(-3, 1):
        # Load the image
        image = cv2.imread(f'./{output_folder}/{test_case}/{iteration}_tone_mapped/{str(i)}.png').astype(np.float32)
        cevr = cv2.imread(f'./{output_folder}/{test_case}/{str(i)}.png').astype(np.float32)
        mask = cv2.imread(f'./{output_folder}/{test_case}/mask.png', cv2.IMREAD_GRAYSCALE)

        cevr = cv2.resize(cevr, (1024, 1024))

        # Convert the image from BGR to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_cevr = cv2.cvtColor(cevr, cv2.COLOR_BGR2YUV)

        # Split the YUV channels
        Y1, U1, V1 = cv2.split(yuv_image)
        Y2, U2, V2 = cv2.split(yuv_cevr)

        residual = np.zeros(Y1.shape, dtype=np.float32)
        residual[Y1 < Y2] = Y2[Y1 < Y2] - Y1[Y1 < Y2]
        residual = cv2.bitwise_and(residual, residual, mask=mask)

        Y1_com = cv2.add(Y1, residual*2)
        Y1_com = np.clip(Y1_com, 0, 255)

        cv2.imwrite(f'./{output_folder}/{test_case}/{iteration}_tone_mapped_residual/residual_{str(i)}.png', residual*10)

        # Merge the channels back together
        yuv_image = cv2.merge([Y1_com, U1, V1])

        # Convert the image back to BGR color space
        output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        # Save or display the modified image
        cv2.imwrite(f'./{output_folder}/{test_case}/{iteration}_tone_mapped_residual/{str(i)}.png', output_image)

        # breakpoint()
        
        new_mask = np.clip((Y1 < Y2) * (mask.reshape(1024, 1024)/255) * 255, 0, 255).astype(np.uint8)    
        # new_mask = cv2.bitwise_and(new_mask, mask, mask=residual)

        cv2.imwrite(f'./{output_folder}/{test_case}/{iteration}_tone_mapped_residual/mask_{str(i)}.png', new_mask)
    
