import numpy as np
import os
from PIL import Image
from diffusers import AutoPipelineForInpainting, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from relighting.pipeline_xlinpaint import CustomStableDiffusionXLControlNetInpaintPipeline, CustomStableDiffusionControlNetInpaintPipeline

from diffusers.utils import load_image
import torch
from transformers import pipeline as transformers_pipeline 

device = "cuda"

img_size = (1024, 1024)

target_folder = "0926_gamma_inverse"  
# test_cases = ["t68", "t78", "t91", "t95"]
test_cases = ["t60"]
iteration = 2

strengths = [1, 0.95, 0.6, 0.2]

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16"
).to(device)


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
extra_kwargs = {"vae": vae} if vae is not None else {}

pipe = CustomStableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    **extra_kwargs,
    ).to("cuda")

for test_case in test_cases:
  os.makedirs(f'./{target_folder}/{test_case}/{str(iteration)}_results/', exist_ok=True)
  # os.makedirs(f'./1016_resizetest/{test_case}/', exist_ok=True)

  img_path = f"./{target_folder}/{test_case}/0.png"
  image = load_image(img_path).resize(img_size)

  mask_path = f"./{target_folder}/{test_case}/mask.png"
  prompt_path = f"./{target_folder}/{test_case}/caption_cog.txt"
  depth_path = f"./{target_folder}/{test_case}/depth.png"

  # prepare mask
  mask_image = load_image(mask_path).resize(img_size)

  # prepare control image
  depth_estimator = transformers_pipeline("depth-estimation", device=device)
  control_image = depth_estimator(image)['depth']
  control_image.save(depth_path)

  # prepare prompt
  with open(prompt_path, "r") as f:
    prompt = f.read()
  prompt = "a clear blue sky with a few white clouds scattered around. The sunlight appears to be shining directly down in the background. The background is very very bright"

  
  for i in range(0, 4):
    generator = torch.Generator(device="cuda")
    exposure = str(-i)
    if iteration == 1:
      img_path = f"./{target_folder}/{test_case}/{exposure}.png"
    else:
      img_path = f"./{target_folder}/{test_case}/{str(iteration-1)}_tone_mapped_residual/{exposure}.png"

    image = load_image(img_path).resize(img_size)
    image_2 = load_image(img_path).resize(img_size)
    # image_2 = load_image(f"./{target_folder}/tone_mapped/{test_case}/{exposure}.png").resize((1024, 1024))

    kwargs = {
          # "prompt_embeds": prompt_embeds,
          "prompt": prompt,
          # 'negative_prompt': args.negative_prompt,
          'num_inference_steps': 1000,
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
          },
          'current_seed': 1000,
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
      img.save(f'./{target_folder}/{test_case}/{str(iteration)}_results/{exposure}.png')
      # img.save(f'./1016_resizetest/{test_case}/{exposure}.png')
