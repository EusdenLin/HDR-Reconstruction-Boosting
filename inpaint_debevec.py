import numpy as np
import os
import cv2
import shutil
import random
import scipy.io
from PIL import Image
from diffusers import AutoPipelineForInpainting, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from relighting.pipeline_xlinpaint import CustomStableDiffusionXLControlNetInpaintPipeline, CustomStableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
import torch
from transformers import pipeline as transformers_pipeline 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# hyperparameters:
# strength, residual, dir, debevec/gamma, prompt, iterations

device = "cuda"
img_size = (1024, 1024)
# results_folder = "results_HDReye"
results_folder = "results_HDReye"
method = "Deep_Recursive_HDRI"
data_folder = "data_HDReye"
output_folder = "results_intermediate_HDReye"
# data_folder = "data_HDReye"
# output_folder = "results_intermediate_HDReye"

# test_cases = ['t81', 't12', 't44', 't71', 't24', 't31', 't13', 't25', 't35', 't69', 't59', 't57', 't4', 't21', 't7', 't38', 't76', 't15', 't30', 't48', 't52', 't1', 't33', 't23', 't37', 't85', 't17', 't82', 't92', 't78', 't6', 't26', 't51', 't41', 't45', 't62', 't65', 't91', 't53', 't87', 't63', 't84', 't39', 't16', 't9', 't90', 't70', 't14', 't83', 't10', 't40', 't32', 't29', 't61', 't46', 't89', 't22', 't2', 't56', 't50', 't74', 't60', 't64', 't47', 't66', 't68', 't55', 't94', 't79', 't72', 't42', 't18', 't54', 't49', 't77', 't43']
# test_cases = ['t81', 't3', 't24', 't13', 't25', 't69', 't7', 't38', 't15', 't5', 't48', 't80', 't28', 't82', 't78', 't73', 't65', 't91', 't11', 't8', 't27', 't9', 't75', 't29', 't46', 't22', 't50', 't60', 't47', 't66', 't68', 't49', 't34', 't77'] ]
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

# times = np.array([1, 1/2, 1/4, 1/8], dtype=np.float32)  

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
  # prompt = "a photo of clear sky. High resuloion image with a lot of details, high luminance and sharpness. 4K, Ultra Quality."
  negative_prompt = "ugly, dark, bad, terrible, awful, horrible, disgusting, gross, nasty, unattractive, unpleasant, repulsive, revolting, vile, foul, abhorrent, loathsome, hideous, unsightly, unlovely, unpleasing, unappealing, uninviting, unwelcome, unattractive, unprepossessing, uncomely, unbeautiful"

  # prepare CRF
  # if not os.path.exists(os.path.join(output_folder, method, test_case, 'inverse_crf.npy')):
  #   images = []
  #   for i in range(4):
  #     images.append(cv2.imread(os.path.join(data_folder, method, test_case, '{}.png'.format(-i))))
  
  #   calibrate = cv2.createCalibrateDebevec()
  #   response = calibrate.process(images, times)
  #   np.save(os.path.join(output_folder, method, test_case, 'inverse_crf.npy'), response)

  #   plt.figure
  #   plt.plot(response.reshape(256, 3)[:, 0], 'r')
  #   plt.plot(response.reshape(256, 3)[:, 1], 'g')
  #   plt.plot(response.reshape(256, 3)[:, 2], 'b')
  #   plt.grid(False)
  #   plt.savefig(os.path.join(output_folder, method, test_case, 'inverse_crf.png'))
  #   plt.close()

  # response = np.load(os.path.join(output_folder, method, test_case, 'inverse_crf.npy'))

  for iteration in range(1, iterations+1):
    os.makedirs(f'./{output_folder}/{method}/{test_case}/{str(iteration)}_results/', exist_ok=True)
    generator = torch.Generator(device="cuda")
    seed = random.randint(0, 1000000)
    print('using random seed:', seed)

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
    
    def readLDR(images_path):
      images = []
      norm_value = 255.0
      n = len(images_path)
      for i in range(n):
          print('Reading image: {}'.format(images_path[i]))
          img = cv2.resize(cv2.cvtColor(cv2.imread(images_path[i], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), img_size)
          img = img.astype(np.float32)
          img = img / norm_value
          images.append(img.copy())

      return images, norm_value

    def weight_fun(img, weight_type, bMeanWeight=0, bounds=[0, 1]):
      weight = np.zeros_like(img)

      if weight_type == 'DEB97':
          Z_min = bounds[0]
          Z_max = bounds[1]
          tr = (Z_min + Z_max) / 2
          delta = Z_max - Z_min
          indx1 = img <= tr
          indx2 = img > tr
          weight[indx1] = img[indx1] - Z_min
          weight[indx2] = Z_max - img[indx2]

          if delta > 0:
              weight = weight / delta


      elif weight_type == 'saturation':
          weight = img

      elif weight_type == 'noise':
          weight = 1 - img

      else:
          raise ValueError('Unknown weight type: {}'.format(weight_type))
      
      np.clip(weight, 0, 1)

      return weight

    def removeCRF(img, response):
      img = img.astype(np.float32)
      col = img.shape[2]
      total_value = response.shape[0]

      delta = 1.0 / (total_value)
      x = np.arange(0, 1, delta)
      imgOut = np.zeros_like(img)

      for i in range(col):   
          interp_func = interp1d(x, response[:, i], kind='nearest', bounds_error=False, fill_value='extrapolate')

          # Apply interpolation to each pixel in channel i
          imgOut[..., i] = interp_func(img[..., i])  # Vectorized computation

      return imgOut


    # path = './data/Deep_/t60'
    # path = f'./data/Deep_Recursive_HDRI/t28'
    inverse_crf = scipy.io.loadmat(f'./{data_folder}/{method}/{test_case}/response.mat')['lin_fun'].reshape((256, 3)).astype(np.float32)

    plt.figure
    plt.plot(inverse_crf.reshape(256, 3)[:, 0], 'r')
    plt.plot(inverse_crf.reshape(256, 3)[:, 1], 'g')
    plt.plot(inverse_crf.reshape(256, 3)[:, 2], 'b')
    plt.grid(False)
    plt.savefig(os.path.join(output_folder, method, test_case, 'inverse_crf.png'))
    plt.close()
    
    images_path = []
    times = np.array([8, 4, 2, 1], dtype=np.float32)  

    images_path.append(f"./{data_folder}/{method}/{test_case}/0.png")
    for i in range(1, 4):
      images_path.append(f'./{output_folder}/{method}/{test_case}/{str(iteration)}_results/{str(-i)}.png')

    images, norm_value = readLDR(images_path)

    delta_value = 1.0 / 65535.0
    scale = 1.0

    i_sat = np.argmin(times)
    i_noise = np.argmax(times)

    threshold = 0.9

    imgOut = np.zeros_like(images[0].astype(np.float32))
    total_weight = np.zeros_like(images[0].astype(np.float32))

    for i in range(len(images)):
      weight_type = 'DEB97'
      tempstack = np.clip(images[i].astype(np.float32) / scale, 0, 1)
      if(i == i_sat):
          if(np.sum(tempstack > threshold) > 0):
              weight_type = 'saturation'

      if(i == i_noise):
          if(np.sum(tempstack < threshold) > 0):
              weight_type = 'noise'

      weight = weight_fun(tempstack, weight_type, 0, [0, 1])
      weight[tempstack < delta_value] = 0 

      # linearize the image
      tempstack = removeCRF(tempstack, inverse_crf)

      dt_i = times[i]

      # merge type: log
      imgOut = imgOut + weight * (np.log(tempstack + delta_value) - np.log(dt_i))
      total_weight = total_weight + weight

    imgOut = imgOut / total_weight
    imgOut = np.exp(imgOut)
    saturation = 1e-4

    # saturation check
    if np.sum(total_weight <= saturation) > 0:
      i_med = np.round(times.shape[0] / 2).astype(np.int8)
      med = np.clip(images[i_med] / scale, 0, 1)
      tempstack = np.clip(images[i_sat].astype(np.float32) / scale, 0, 1)

      img_sat = removeCRF(tempstack, inverse_crf)
      img_sat = img_sat / times[i_sat]

      mask = np.ones_like(total_weight).astype(np.int8)
      mask[total_weight > saturation] = 0
      mask[med > 0.5] = 0
      mask = np.expand_dims(mask, axis=2)

      if np.max(mask) > 0.5:
          print('saturation')

          for i in range(3):
              io_i = imgOut[:, :, i]
              is_i = img_sat[:, :, i]
              io_i[mask] = is_i[mask]
              imgOut[:, :, i] = io_i
      
    hdr = imgOut

    os.makedirs(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped/', exist_ok=True)
    cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped/hdr.hdr', hdr)
    inverse_crf = inverse_crf.reshape((256, 3)) 

    ldr_indices = np.arange(256)

    for i in range(0, 3):
      hdr_scaled = hdr * pow(2, i)

      tone_mapped_image = np.zeros_like(hdr)  # Prepare output image
      for c in range(3):  # Loop over RGB channels
          # Create an interpolation function for the channel
          interp_func = interp1d(inverse_crf[:, c], ldr_indices, kind='linear', bounds_error=False, fill_value="extrapolate")
          
          # Apply the interpolation to map HDR to LDR
          tone_mapped_image[..., c] = interp_func(hdr_scaled[..., c])

      ldr = np.clip(tone_mapped_image, 0, 255).astype(np.uint8)
      ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
      cv2.imwrite(f'./{output_folder}/{method}/{test_case}/{iteration}_tone_mapped/{str(i-3)}.png', ldr)

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
  # shutil.copy(f'./{data_folder}/{method}/{test_case}/0.png', f'./{results_folder}/{method}/{test_case}/inpaint/-0.png')
  # shutil.copy(f'./{data_folder}/{method}/{test_case}/0.png', f'./{results_folder}/{method}/{test_case}/baseline/-0.png')
  
    
