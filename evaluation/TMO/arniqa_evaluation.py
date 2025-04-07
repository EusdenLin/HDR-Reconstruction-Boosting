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
path = './results_HDReye/KK_TMO/gamma_WACV'
targer_folder = './results_HDReye/KK_TMO/gamma_WACV'
test_cases = os.listdir(f"./{path}")

remake_cases = []

score_inpaint = 0
score_baseline = 0

for test_case in test_cases:
    print(test_case)
    # for i in range(1, 4):
    inpaint_path = f"./{targer_folder}/{test_case}/inpaint.png"
    img = Image.open(inpaint_path).convert("RGB")
    img = transforms.Resize((1024, 1024))(img)
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    img = preprocess(img).unsqueeze(0).to(device)
    img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    score_inpaint += score.item()

    score_temp = score.item()

    baseline_path = f"./{path}/{test_case}/baseline.png"
    img = Image.open(baseline_path).convert("RGB")
    img = transforms.Resize((1024, 1024))(img)
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    img = preprocess(img).unsqueeze(0).to(device)
    img_ds = preprocess(img_ds).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    score_baseline += score.item()

    if score_temp < score.item() - 0.01:
        print(f"Case {test_case} is worse than baseline")
        print(score_temp, score.item())
        remake_cases.append(test_case)

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

print(f"Average image quality score of inpaint: {score_inpaint / (len(test_cases) )}")
print(f"Average image quality score of baseline: {score_baseline / (len(test_cases))}")

print(remake_cases)
# print(f"Average image quality score of gamma: {score_gamma / (len(test_cases))}")
# print(f"Average image quality score of glowgan: {score_glow / (len(test_cases))}")

'''
CEVR ['C32', 'C45', 'C33', 'C6', 'C17', 'C31', 'C23', 'C36', 'C12', 'C21', 'C13', 'C14', 'C10']
gamma ['C7', 'C44', 'C32', 'C39', 'C19', 'C45', 'C41', 'C33', 'C46', 'C6', 'C42', 'C30', 'C22', 'C23', 'C5', 'C24', 'C8', 'C12', 'C37', 'C29', 'C25', 'C13', 'C43', 'C38', 'C26', 'C14', 'C10']
'''