import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from relighting.pipeline_xlinpaint import CustomStableDiffusionXLControlNetInpaintPipeline, CustomStableDiffusionControlNetInpaintPipeline

from diffusers.utils import load_image
import torch
from transformers import pipeline as transformers_pipeline 

device = "cuda"

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

# controlnet = ControlNetModel.from_pretrained(
#     "thibaud/controlnet-sd21-normalbae-diffusers",
#     torch_dtype=torch.float16,
# ).to(device)

# pipe = CustomStableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     controlnet=controlnet,
#     torch_dtype=torch.float16,
#     ).to("cuda")



# [t60, t68, t91]
test_case = "t60"
exposure = "-3"

img_path = "./data/strength_test/"+test_case+"/val/EV"+exposure+".png"
mask_path = "./data/strength_test/"+test_case+"/val/mask_val.png"
# prompt_path = "./data/strength_test/"+test_case+"/caption.txt"

image = load_image(img_path).resize((1024, 1024))
mask_image = load_image(mask_path).resize((1024, 1024))
depth_estimator = transformers_pipeline("depth-estimation", device=device)
control_image = depth_estimator(image)['depth']
control_image.save(f"./test.png")

prompt = "cloudy sky in the background"
generator = torch.Generator(device="cuda")

kwargs = {
      # "prompt_embeds": prompt_embeds,
      "prompt": prompt,
      # 'negative_prompt': args.negative_prompt,
      'num_inference_steps': 200,
      'generator': generator,
      'image': image,
      'mask_image': mask_image,
      'control_image': control_image,
      'strength': 0.8,
      'current_seed': 1000, # we still need seed in the pipeline!
      'controlnet_conditioning_scale': 0.5,
      'height': 1024,
      'width': 1024,
      'guidance_scale': 5.0,
  }


image = pipe(
  **kwargs
).images

for i, img in enumerate(image):
  img.save(f"./data/strength_test/"+test_case+"/results/"+exposure+".png")
