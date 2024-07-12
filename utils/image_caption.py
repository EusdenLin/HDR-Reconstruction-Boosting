from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

folder_path = "./data/VDS"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT

for image_path in os.listdir(folder_path):
    if image_path != 't60':
        continue
    path = os.path.join(folder_path, image_path, f'{image_path}_0EV_true.jpg.png')

    image = Image.open(path)

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(image_path)
    print(generated_text)

    # store the generated text in a file
    with open(os.path.join(folder_path, image_path, 'caption.txt'), 'w') as f:
        f.write(generated_text)
        f.close()
