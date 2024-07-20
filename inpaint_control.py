import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from relighting.pipeline_xlinpaint import CustomStableDiffusionXLControlNetInpaintPipeline

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

# [t60, t68, t91]
test_case = "t60"

img_path = "./data/strength_test/"+test_case+"/test/-1.png"
mask_path = "./data/strength_test/"+test_case+"/val/mask_val.png"
prompt_path = "./data/strength_test/"+test_case+"/caption.txt"

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
      'strength': 0.85,
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
  img.save(f"./data/strength_test/"+test_case+"/results/-1.png")



# def estimate_scene_depth(image, depth_estimator):
#     #image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
#     #with torch.no_grad(), torch.autocast("cuda"):
#     #    depth_map = depth_estimator(image).predicted_depth

#     depth_map = depth_estimator(image)['predicted_depth']
#     W, H = image.size
#     depth_map = torch.nn.functional.interpolate(
#         depth_map.unsqueeze(1),
#         size=(H, W),
#         mode="bicubic",
#         align_corners=False,
#     )
#     depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_map = (depth_map - depth_min) / (depth_max - depth_min)
#     image = torch.cat([depth_map] * 3, dim=1)

#     image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
#     image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
#     return image

# class ControlSignalGenerator():
#     def __init__(self, sd_arch, control_signal_type, device):
#         self.sd_arch = sd_arch
#         self.control_signal_type = control_signal_type
#         self.device = device

#     def estimate_scene_depth(image, depth_estimator):
#         #image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
#         #with torch.no_grad(), torch.autocast("cuda"):
#         #    depth_map = depth_estimator(image).predicted_depth

#         depth_map = depth_estimator(image)['predicted_depth']
#         W, H = image.size
#         depth_map = torch.nn.functional.interpolate(
#             depth_map.unsqueeze(1),
#             size=(H, W),
#             mode="bicubic",
#             align_corners=False,
#         )
#         depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
#         depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
#         depth_map = (depth_map - depth_min) / (depth_max - depth_min)
#         image = torch.cat([depth_map] * 3, dim=1)

#         image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
#         image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
#         return image

#     def process_sdxl_depth(self, input_image, normal_ball=None, mask_ball=None, x=None, y=None, r=None):
#         DEPTH_ESTIMATOR = "Intel/dpt-hybrid-midas"
#         if getattr(self, 'depth_estimator', None) is None:
#             self.depth_estimator = transformers_pipeline("depth-estimation", model=DEPTH_ESTIMATOR, device=self.device.index)

#         control_image = estimate_scene_depth(input_image, depth_estimator=self.depth_estimator)
#         xs = [x] if not isinstance(x, list) else x
#         ys = [y] if not isinstance(y, list) else y
#         rs = [r] if not isinstance(r, list) else r
        
#         return control_image


#     def __call__(self, *args, **kwargs):
#         process_fn = getattr(self, f"process_{self.sd_arch}_{self.control_signal_type}", None)
#         if process_fn is None:
#             raise ValueError
#         else:
#             return process_fn(*args, **kwargs)

# class Inpainter():
#     def __init__(self, pipeline, sd_arch, control_generator, disable_water_mask=True):
#         self.pipeline = pipeline
#         self.sd_arch = sd_arch
#         self.control_generator = control_generator
#         self.median = {}

#     @classmethod
#     def from_sdxl(cls, 
#                 model, 
#                 controlnet=None, 
#                 device=0, 
#                 sampler="unipc", 
#                 torch_dtype=torch.float16,
#                 disable_water_mask=True,
#                 use_fixed_vae=True,
#                 offload=False
#     ):
#         vae = VAE_MODELS["sdxl"]
#         vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch_dtype).to(device) if use_fixed_vae else None
#         extra_kwargs = {"vae": vae} if vae is not None else {}
        
#         if controlnet is not None:
#             control_signal_type = get_control_signal_type(controlnet)
#             controlnet = ControlNetModel.from_pretrained(
#                 controlnet,
#                 variant="fp16" if torch_dtype == torch.float16 else None,
#                 use_safetensors=True,
#                 torch_dtype=torch_dtype,
#             ).to(device)
#             pipe = CustomStableDiffusionXLControlNetInpaintPipeline.from_pretrained(
#                 model,
#                 controlnet=controlnet,
#                 variant="fp16" if torch_dtype == torch.float16 else None,
#                 use_safetensors=True,
#                 torch_dtype=torch_dtype,
#                 **extra_kwargs,
#             ).to(device)
#             control_generator = ControlSignalGenerator("sdxl", control_signal_type, device=device)
#         else:
#             pipe = CustomStableDiffusionXLInpaintPipeline.from_pretrained(
#                 model,
#                 variant="fp16" if torch_dtype == torch.float16 else None,
#                 use_safetensors=True,
#                 torch_dtype=torch_dtype,
#                 **extra_kwargs,
#             ).to(device)
#             control_generator = None
        
#         try:
#             if torch_dtype==torch.float16 and device != torch.device("cpu"):
#                 pipe.enable_xformers_memory_efficient_attention()
#         except:
#             pass
        
#         if offload and device != torch.device("cpu"):
#             pipe.enable_model_cpu_offload()
#         pipe.set_progress_bar_config(disable=True)
#         pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
        
#         return BallInpainter(pipe, "sdxl", control_generator, disable_water_mask)

#     # TODO: this method should be replaced by inpaint(), but we'll leave it here for now
#     # otherwise, the existing experiment code will break down
#     def __call__(self, *args, **kwargs):
#         return self.pipeline(*args, **kwargs)

#     def _default_height_width(self, height=None, width=None):
#         if (height is not None) and (width is not None):
#             return height, width
#         if self.sd_arch == "sd":
#             return (512, 512)
#         elif self.sd_arch == "sdxl":
#             return (1024, 1024)
#         else:
#             raise NotImplementedError

#     # this method is for sanity check only
#     def get_cache_control_image(self):
#         control_image = getattr(self, "cache_control_image", None)
#         return control_image

#     def prepare_control_signal(self, image, controlnet_conditioning_scale, extra_kwargs):
#         if self.control_generator is not None:
#             control_image = self.control_generator(image, **extra_kwargs)
#             controlnet_kwargs = {
#                 "control_image": control_image,
#                 "controlnet_conditioning_scale": controlnet_conditioning_scale
#             }
#             self.cache_control_image = control_image
#         else:
#             controlnet_kwargs = {}

#         return controlnet_kwargs

#     def get_cache_median(self, it):
#         if it in self.median: return self.median[it]
#         else: return None

#     def reset_median(self):
#         self.median = {}
#         print("Reset median")

#     def load_median(self, path):
#         if os.path.exists(path):
#             with open(path, "rb") as f:
#                 self.median = pickle.load(f)
#                 print(f"Loaded median from {path}")
#         else:
#             print(f"Median not found at {path}!")

#     def inpaint_iterative(
#         self,
#         prompt=None,
#         negative_prompt="",
#         num_inference_steps=30,
#         generator=None, # TODO: remove this
#         image=None,
#         mask_image=None,
#         height=None,
#         width=None,
#         controlnet_conditioning_scale=0.5,
#         num_images_per_prompt=1,
#         current_seed=0,
#         cross_attention_kwargs={},
#         strength=0.8,
#         num_iteration=2,
#         ball_per_iteration=30,
#         agg_mode="median",
#         save_intermediate=True,
#         cache_dir="./temp_inpaint_iterative",
#         disable_progress=False,
#         prompt_embeds=None,
#         pooled_prompt_embeds=None,
#         use_cache_median=False,
#         guidance_scale=5.0, # In the paper, we use guidance scale to 5.0 (same as pipeline_xl.py)
#         **extra_kwargs,
#     ):
#         def computeMedian(ball_images):
#             all = np.stack(ball_images, axis=0)
#             median = np.median(all, axis=0)
#             idx_median = np.argsort(all, axis=0)[all.shape[0]//2]
#             # print(all.shape)
#             # print(idx_median.shape)
#             return median, idx_median

#         def generate_balls(avg_image, current_strength, ball_per_iteration, current_iteration):
#             print(f"Inpainting balls for {current_iteration} iteration...")
#             controlnet_kwargs = self.prepare_control_signal(
#                 image=avg_image,
#                 controlnet_conditioning_scale=controlnet_conditioning_scale,
#                 extra_kwargs=extra_kwargs,
#             )

#             ball_images = []
#             for i in tqdm(range(ball_per_iteration), disable=disable_progress):
#                 seed = current_seed + i
#                 new_generator = torch.Generator().manual_seed(seed)
#                 output_image = self.pipeline(
#                     prompt=prompt,
#                     negative_prompt=negative_prompt,
#                     num_inference_steps=num_inference_steps,
#                     generator=new_generator,
#                     image=avg_image,
#                     mask_image=mask_image,
#                     height=height,
#                     width=width,
#                     num_images_per_prompt=num_images_per_prompt,
#                     strength=current_strength,
#                     newx=x,
#                     newy=y,
#                     newr=r,
#                     current_seed=seed,
#                     cross_attention_kwargs=cross_attention_kwargs,
#                     prompt_embeds=prompt_embeds,
#                     pooled_prompt_embeds=pooled_prompt_embeds,
#                     guidance_scale=guidance_scale,
#                     **controlnet_kwargs
#                 ).images[0]
                
#                 ball_image = crop_ball(output_image, mask_ball_for_crop, x, y, r)
#                 ball_images.append(ball_image)

#                 if save_intermediate:
#                     os.makedirs(os.path.join(cache_dir, str(current_iteration)), mode=0o777, exist_ok=True)
#                     output_image.save(os.path.join(cache_dir, str(current_iteration), f"raw_{i}.png"))
#                     Image.fromarray(ball_image).save(os.path.join(cache_dir, str(current_iteration), f"ball_{i}.png"))
#                     # chmod 777
#                     os.chmod(os.path.join(cache_dir, str(current_iteration), f"raw_{i}.png"), 0o0777)
#                     os.chmod(os.path.join(cache_dir, str(current_iteration), f"ball_{i}.png"), 0o0777)

            
#             return ball_images

#         if save_intermediate:
#             os.makedirs(cache_dir, exist_ok=True)

#         height, width = self._default_height_width(height, width)

#         x = extra_kwargs["x"]
#         y = extra_kwargs["y"]
#         r = 256  if "r" not in extra_kwargs else extra_kwargs["r"]
#         _, mask_ball_for_crop = get_ideal_normal_ball(size=r)
        
#         # generate initial average ball
#         avg_image = image
#         ball_images = generate_balls(
#             avg_image,
#             current_strength=1.0,
#             ball_per_iteration=ball_per_iteration,
#             current_iteration=0,
#         )

#         # ball refinement loop
#         image = np.array(image)
#         for it in range(1, num_iteration+1):
#             if use_cache_median and (self.get_cache_median(it) is not None):
#                 print("Use existing median")
#                 all = np.stack(ball_images, axis=0)
#                 idx_median = self.get_cache_median(it)
#                 avg_ball = all[idx_median, 
#                     np.arange(idx_median.shape[0])[:, np.newaxis, np.newaxis],
#                     np.arange(idx_median.shape[1])[np.newaxis, :, np.newaxis],
#                     np.arange(idx_median.shape[2])[np.newaxis, np.newaxis, :]
#                 ]
#             else:
#                 avg_ball, idx_median = computeMedian(ball_images)
#                 print("Add new median")
#                 self.median[it] = idx_median
            
#             avg_image = merge_normal_map(image, avg_ball, mask_ball_for_crop, x, y)
#             avg_image = Image.fromarray(avg_image.astype(np.uint8))
#             if save_intermediate:
#                 avg_image.save(os.path.join(cache_dir, f"average_{it}.png"))
#                 # chmod777
#                 os.chmod(os.path.join(cache_dir, f"average_{it}.png"), 0o0777)
            
#             ball_images = generate_balls(
#                 avg_image,
#                 current_strength=strength,
#                 ball_per_iteration=ball_per_iteration if it < num_iteration else 1,
#                 current_iteration=it,
#             )

#         # TODO: add algorithm for select the best ball
#         best_ball = ball_images[0]
#         output_image = merge_normal_map(image, best_ball, mask_ball_for_crop, x, y)
#         return Image.fromarray(output_image.astype(np.uint8))

#     def inpaint(
#         self,
#         prompt=None,
#         negative_prompt=None,
#         num_inference_steps=30,
#         generator=None,
#         image=None,
#         mask_image=None,
#         height=None,
#         width=None,
#         controlnet_conditioning_scale=0.5,
#         num_images_per_prompt=1,
#         strength=1.0,
#         current_seed=0,
#         cross_attention_kwargs={},
#         prompt_embeds=None,
#         pooled_prompt_embeds=None,
#         guidance_scale=5.0, # (same as pipeline_xl.py)
#         **extra_kwargs,
#     ):
#         height, width = self._default_height_width(height, width)

#         controlnet_kwargs = self.prepare_control_signal(
#             image=image,
#             controlnet_conditioning_scale=controlnet_conditioning_scale,
#             extra_kwargs=extra_kwargs,
#         )
        
#         if generator is None:
#             generator = torch.Generator().manual_seed(0)

#         output_image = self.pipeline(
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             num_inference_steps=num_inference_steps,
#             generator=generator,
#             image=image,
#             mask_image=mask_image,
#             height=height,
#             width=width,
#             num_images_per_prompt=num_images_per_prompt,
#             strength=strength,
#             newx = extra_kwargs["x"],
#             newy = extra_kwargs["y"],
#             newr = getattr(extra_kwargs, "r", 256), # default to ball_size = 256
#             current_seed=current_seed,
#             cross_attention_kwargs=cross_attention_kwargs,
#             prompt_embeds=prompt_embeds,
#             pooled_prompt_embeds=pooled_prompt_embeds,
#             guidance_scale=guidance_scale,
#             **controlnet_kwargs
#         )

#         return output_image