import numpy as np
import os
import cv2
import shutil
import random
from PIL import Image
from diffusers import AutoPipelineForInpainting, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from relighting.pipeline_xlinpaint import CustomStableDiffusionXLControlNetInpaintPipeline, CustomStableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
import torch
from transformers import pipeline as transformers_pipeline 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

device = "cuda"
img_size = (1024, 1024)
results_folder = "results"
method = "gamma"

data_folder = "data"
output_folder = "results_intermediate"
test_cases = None
iterations = 4

strengths = [1, 0.95, 0.95, 0.95, 0.95]

if test_cases is None:
  test_cases = os.listdir(f"./{data_folder}/{method}")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16"
).to(device)

times = np.array([1, 1/2, 1/4, 1/8], dtype=np.float32)  

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

  # prepare prompt
  with open(prompt_path, "r") as f:
    prompt = f.read()
  prompt = "a photo of shiny sky with a few white clouds scattered around. The background is very very bright"
  negative_prompt = "ugly, dark, bad, terrible, awful, horrible, disgusting, gross, nasty, unattractive, unpleasant, repulsive, revolting, vile, foul, abhorrent, loathsome, hideous, unsightly, unlovely, unpleasing, unappealing, uninviting, unwelcome, unattractive, unprepossessing, uncomely, unbeautiful"

  # prepare CRF
  if not os.path.exists(os.path.join(output_folder, method, test_case, 'inverse_crf.npy')):
    images = []
    for i in range(4):
      images.append(cv2.imread(os.path.join(data_folder, method, test_case, '{}.png'.format(-i))))
  
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, times)
    np.save(os.path.join(output_folder, method, test_case, 'inverse_crf.npy'), response)

    plt.figure
    plt.plot(response.reshape(256, 3)[:, 0], 'r')
    plt.plot(response.reshape(256, 3)[:, 1], 'g')
    plt.plot(response.reshape(256, 3)[:, 2], 'b')
    plt.grid(False)
    plt.savefig(os.path.join(output_folder, method, test_case, 'inverse_crf.png'))
    plt.close()

  response = np.load(os.path.join(output_folder, method, test_case, 'inverse_crf.npy'))

  for iteration in range(1, iterations+1):
    os.makedirs(f'./{output_folder}/{method}/{test_case}/{str(iteration)}_results/', exist_ok=True)
    generator = torch.Generator(device="cuda")
    seed = random.randint(0, 1000000)
    print(seed)

    for i in range(1, 4):
      generator = torch.Generator(device="cuda")
      generator.manual_seed(seed)
      exposure = str(-i)
      if iteration == 1:
        img_path = f"./{data_folder}/{method}/{test_case}/{exposure}.png"
      else:
        img_path = f"./{output_folder}/{method}/{test_case}/{str(iteration-1)}_tone_mapped_residual/{exposure}.png"
        mask_path = f"./{output_folder}/{method}/{test_case}/{str(iteration-1)}_tone_mapped_residual/mask_{str(-i)}.png"
        

      mask_image = load_image(mask_path).resize(img_size)
      image = load_image(img_path).resize(img_size)

      kwargs = {
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
    os.makedirs(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped/', exist_ok=True)
    images = []
    images.append(cv2.resize(cv2.imread(f"./{data_folder}/{method}/{test_case}/0.png"), img_size))
    for i in range(1, 4):
      images.append(cv2.imread(f'./{output_folder}/{method}/{test_case}/{str(iteration)}_results/{str(-i)}.png'))
    
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times, response)
    cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped/hdr.hdr', hdr)

    hdr /= 10

    ldr_indices = np.arange(256)
    # Tone mapping
    for i in range(-3, 0):
      reverse_CRF = response.reshape((256, 3))/10
      hdr_rescaled = hdr * (1/np.power(2, -i))
      tone_mapped_image = np.zeros_like(hdr)
      for c in range(3):
          interp_func = interp1d(reverse_CRF[:, c], ldr_indices, kind='linear', bounds_error=False, fill_value="extrapolate")
          tone_mapped_image[..., c] = interp_func(hdr_rescaled[..., c])

      ldr = np.clip(tone_mapped_image, 0, 255).astype(np.uint8)
      cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped/{str(i)}.png', ldr)

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

        Y1_com = cv2.add(Y1, residual*2)
        Y1_com = np.clip(Y1_com, 0, 255)

        cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped_residual/residual_{str(i)}.png', residual*10)

        # Merge the channels back together
        yuv_image = cv2.merge([Y1_com, U1, V1])

        # Convert the image back to BGR color space
        output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped_residual/{str(i)}.png', output_image)

        new_mask = np.clip((Y1 < Y2) * (mask.reshape(1024, 1024)/255) * 255, 0, 255).astype(np.uint8)    

        cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped_residual/mask_{str(i)}.png', new_mask)
  
  # Save to results
  os.makedirs(f'./{results_folder}/{method}/{test_case}/baseline', exist_ok=True)
  os.makedirs(f'./{results_folder}/{method}/{test_case}/inpaint', exist_ok=True)

  for i in range(1, 4):
    shutil.copy(f'./{output_folder}/{method}/{test_case}/{str(iterations)}_tone_mapped/{str(-i)}.png', f'./{results_folder}/{method}/{test_case}/inpaint/{str(-i)}.png')
    shutil.copy(f'./{data_folder}/{method}/{test_case}/{str(-i)}.png', f'./{results_folder}/{method}/{test_case}/baseline/{str(-i)}.png')
  
  shutil.copy(f'./{data_folder}/{method}/{test_case}/0.png', f'./{results_folder}/{method}/{test_case}/inpaint/-0.png')
  shutil.copy(f'./{data_folder}/{method}/{test_case}/0.png', f'./{results_folder}/{method}/{test_case}/baseline/-0.png')
  
    
