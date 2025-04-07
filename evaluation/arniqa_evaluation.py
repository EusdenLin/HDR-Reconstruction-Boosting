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
path = 'results/Deep_Recursive_HDRI'
targer_folder = f"results/gamma"
test_cases = os.listdir(f"./{path}")

remake_case = []

score_inpaint = 0
score_baseline = 0

# skipped_cases = ['t9', 't8', 't29', 't66', 't73', 't80', 't25', 't13', 't65']
skipped_cases = []
print('total length: ', len(test_cases))
for test_case in test_cases:
    # print(test_case)
    if test_case in skipped_cases:
        continue
    ev_score_inpaint = 0
    ev_score_baseline = 0
    for i in range(1, 4):
        inpaint_path = f"./{targer_folder}/{test_case}/inpaint/{str(-i)}.png"
        img = Image.open(inpaint_path).convert("RGB")
        img = transforms.Resize((1024, 1024))(img)
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
        img = preprocess(img).unsqueeze(0).to(device)
        img_ds = preprocess(img_ds).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            score = model(img, img_ds, return_embedding=False, scale_score=True)
        score_inpaint += score.item()
        ev_score_inpaint += score.item()
        # print(score.item())

        baseline_path = f"./{targer_folder}/{test_case}/baseline/{str(-i)}.png"
        img = Image.open(baseline_path).convert("RGB")
        img = transforms.Resize((1024, 1024))(img)
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
        img = preprocess(img).unsqueeze(0).to(device)
        img_ds = preprocess(img_ds).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            score = model(img, img_ds, return_embedding=False, scale_score=True)
        score_baseline += score.item()
        ev_score_baseline += score.item()
        # print(score.item())
    
    if ev_score_inpaint < ev_score_baseline -0.03:
        print(test_case)
        print(ev_score_inpaint / 3, ev_score_baseline / 3)
        remake_case.append(test_case) 

    # gamma_path = f"./{path}/{test_case}/gamma.png"
    # img = Image.open(gamma_path).convert("RGB")
    # img = transforms.Resize((256, 256))(img)
    # img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    # img = preprocess(img).unsqueeze(0).to(device)
    # img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    # with torch.no_grad(), torch.cuda.amp.autocast():
    #     score = model(img, img_ds, return_embedding=False, scale_score=True)
    # score_gamma += score.item()

    # glow_path = f"./{path}/{test_case}/glowgan.png"
    # img = Image.open(glow_path).convert("RGB")
    # img = transforms.Resize((256, 256))(img)
    # img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    # img = preprocess(img).unsqueeze(0).to(device)
    # img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    # with torch.no_grad(), torch.cuda.amp.autocast():
    #     score = model(img, img_ds, return_embedding=False, scale_score=True)
    # score_glow += score.item()

print(f"Average image quality score of inpaint: {score_inpaint / (len(test_cases) - len(skipped_cases)) / 3}")
print(f"Average image quality score of baseline: {score_baseline / (len(test_cases) - len(skipped_cases)) / 3}")
print(remake_case, len(remake_case))

# only testing dataset:
# CEVR: 0.234
# CEVR+Ours: 0.242

# Deep_Recursive_HDRI: 0.285
# Deep_Recursive_HDRI+Ours: 0.291


