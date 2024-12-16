import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load the model
model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                       regressor_dataset="kadid10k")    # You can choose any of the available datasets
model.eval().to(device)

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# path = '1119_diversity'
path = 'hdr_images'
targer_folder = f"./hdr_images"
test_cases = os.listdir(f"./{path}")

score_gamma = 0
score_inpaint = 0
score_inpaint_turbo = 0
score_glow = 0

for test_case in test_cases:
    print(test_case)
    # for i in range(1, 4):
    inpaint_path = f"./{targer_folder}/{test_case}/inpaint.png"
    img = Image.open(inpaint_path).convert("RGB")
    img = transforms.Resize((256, 256))(img)
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    img = preprocess(img).unsqueeze(0).to(device)
    img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    score_inpaint += score.item()
    print(f"Image quality score of {test_case}: {score.item()}")

    inpaint_path_turbo = f"./{path}/{test_case}/inpaint_turbo.png"
    img = Image.open(inpaint_path_turbo).convert("RGB")
    img = transforms.Resize((256, 256))(img)
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    img = preprocess(img).unsqueeze(0).to(device)
    img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    score_inpaint_turbo += score.item()

    gamma_path = f"./{path}/{test_case}/gamma.png"
    img = Image.open(gamma_path).convert("RGB")
    img = transforms.Resize((256, 256))(img)
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    img = preprocess(img).unsqueeze(0).to(device)
    img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    score_gamma += score.item()

    glow_path = f"./{path}/{test_case}/glowgan.png"
    img = Image.open(glow_path).convert("RGB")
    img = transforms.Resize((256, 256))(img)
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    img = preprocess(img).unsqueeze(0).to(device)
    img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    score_glow += score.item()

print(f"Average image quality score of inpaint: {score_inpaint / (len(test_cases))}")
print(f"Average image quality score of inpaint_turbo: {score_inpaint_turbo / (len(test_cases))}")
print(f"Average image quality score of gamma: {score_gamma / (len(test_cases))}")
print(f"Average image quality score of glowgan: {score_glow / (len(test_cases))}")
