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

target_folder = "1021_multi_mask"  
output_folder = "1021_multi_mask"
test_cases = ["t60", "t68", "t78", "t91", "t95"]
# test_cases = ["t60"]
iterations = 4

strengths = [1, 0.95, 0.95, 0.95, 0.95]

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16"
).to(device)


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
extra_kwargs = {"vae": vae} if vae is not None else {}

for test_case in test_cases:

  # os.makedirs(f'./1016_resizetest/{test_case}/', exist_ok=True)
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
  depth_estimator = transformers_pipeline("depth-estimation", device=device)
  control_image = depth_estimator(image)['depth']
  control_image.save(depth_path)

  # prepare prompt
  with open(prompt_path, "r") as f:
    prompt = f.read()
  prompt = "a clear blue sky with a few white clouds scattered around. The sunlight appears to be shining directly down in the background. The background is very very bright"

  for iteration in range(1, iterations+1):
    os.makedirs(f'./{output_folder}/{test_case}/{str(iteration)}_results/', exist_ok=True)
    for i in range(0, 4):

      generator = torch.Generator(device="cuda")
      if iteration > 1:
        generator.manual_seed(random.randint(0, 1000000))

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
            'num_inference_steps': 200,
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
            'current_seed': random.randint(0, 1000000),
            'controlnet_conditioning_scale': 0.5,
            'height': 1024,
            'width': 1024,
            'guidance_scale': 5.0,
        }

      with torch.no_grad():
        image = pipe(
          **kwargs
        ).images

      for i, img in enumerate(image):
        img.save(f'./{output_folder}/{test_case}/{str(iteration)}_results/{exposure}.png')
        # img.save(f'./1016_resizetest/{test_case}/{exposure}.png')

    # Merge to HDR

    command = [
      "python", "exposure2hdr.py",
      "--input_dir", f"{output_folder}/{test_case}/{str(iteration)}_results",
      "--output_dir", f"{output_folder}/{test_case}/",
      "--iteration", f"{str(iteration)}"
    ]
    subprocess.run(command)

    # Residual
    os.makedirs(f'./{output_folder}/{test_case}/{iteration}_tone_mapped_residual/', exist_ok=True)
    for i in range(-3, 1):
        # Load the image
        image = cv2.imread(f'./{output_folder}/{test_case}/{iteration}_tone_mapped/{str(i)}.png')
        cevr = cv2.imread(f'./{output_folder}/{test_case}/{str(i)}.png')
        mask = cv2.imread(f'./{output_folder}/{test_case}/mask.png', cv2.IMREAD_GRAYSCALE)

        cevr = cv2.resize(cevr, (1024, 1024))

        def gamma_correction(image, gamma=2.2):
            return np.power(image / 255.0, gamma) * 255.0

        image = gamma_correction(image)
        cevr = gamma_correction(cevr)

        # Convert the image from BGR to YUV color space
        yuv_image = gamma_correction(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
        yuv_cevr = gamma_correction(cv2.cvtColor(cevr, cv2.COLOR_BGR2YUV))

        # Split the YUV channels
        Y1, U1, V1 = cv2.split(yuv_image)
        Y2, U2, V2 = cv2.split(yuv_cevr)

        residual = np.zeros(Y1.shape, dtype=np.uint8)
        residual[Y1 < Y2] = Y2[Y1 < Y2] - Y1[Y1 < Y2]
        residual = cv2.bitwise_and(residual, residual, mask=mask)

        Y1_com = cv2.add(Y1, residual*3)
        Y1_com = np.clip(Y1_com, 0, 255)

        # Merge the channels back together
        yuv_image = cv2.merge([Y1_com, U1, V1])

        # Convert the image back to BGR color space
        output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        output_image = gamma_correction(output_image, 1/2.2)

        # Save or display the modified image
        cv2.imwrite(f'./{output_folder}/{test_case}/{iteration}_tone_mapped_residual/{str(i)}.png', output_image)

        # breakpoint()
        new_mask = np.clip((Y1 < Y2) * (mask.reshape(1024, 1024)/255) * 255, 0, 255).astype(np.uint8)    
        # new_mask = cv2.bitwise_and(new_mask, mask, mask=residual)

        cv2.imwrite(f'./{output_folder}/{test_case}/{iteration}_tone_mapped_residual/mask_{str(i)}.png', new_mask)
    
