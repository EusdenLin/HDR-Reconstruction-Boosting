from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")
control = pipe.control

# [t60, t68, t91]
test_case = "t60"

img_path = "./data/VDS/"+test_case+"/val/EV-3.png"
mask_path = "./data/VDS/"+test_case+"/val/mask_val.png"
prompt_path = "./data/VDS/"+test_case+"/caption.txt"

image = load_image(img_path).resize((1024, 1024))
mask_image = load_image(mask_path).resize((1024, 1024))

prompt = "cloudy sky in the background"
generator = torch.Generator(device="cuda")

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=5.0,
  num_inference_steps=200,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images

for i, img in enumerate(image):
  img.save(f"./test/"+test_case+".png")
