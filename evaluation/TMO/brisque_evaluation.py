from brisque import BRISQUE
import cv2
import os
import pandas as pd
import math 

def get_score(img_paths, inpaint_paths):
    avg_score_baseline = 0
    avg_score_inpaint = 0
    dataframe = pd.read_csv('result.csv')
    for i in range(len(img_paths)):
        
        img_baseline = cv2.imread(img_paths[i])
        img_inpaint = cv2.imread(inpaint_paths[i])
        img_baseline = cv2.resize(img_baseline, (1024, 1024))
        img_inpaint = cv2.resize(img_inpaint, (1024, 1024))
        obj = BRISQUE(url=False)
        score_baseline = obj.score(img_baseline)
        score_inpaint = obj.score(img_inpaint)
        print(int(os.path.basename(inpaint_paths[i].replace('/inpaint.png', '')).replace('t', '')))
        dataframe.iloc[int(os.path.basename(inpaint_paths[i].replace('/inpaint.png', '')).replace('t', ''))-1]['brisque'] = score_inpaint

        if math.isnan(score_baseline):
            avg_score_baseline += 100
        else:
            avg_score_baseline += score_baseline

        if math.isnan(score_inpaint):
            avg_score_inpaint += 100
        else:
            avg_score_inpaint += score_inpaint
            
        if score_inpaint - 1 > score_baseline:
            print(img_paths[i], inpaint_paths[i])
            print(f'Baseline: {score_baseline}, Inpaint: {score_inpaint}')

    avg_score_baseline /= len(img_paths)
    avg_score_inpaint /= len(img_paths)
    dataframe.to_csv('results.csv', index=False)
    return avg_score_baseline, avg_score_inpaint
# target_folder = './evaluation_RH'
# target_folder = './evaluation_single'
RH_folder = '/home/ytlin/boosting_HDR/results/VDS/RH_TMO/CEVR'
case_folder = '/home/ytlin/boosting_HDR/results/VDS/KK_TMO/CEVR'

inpaint_paths = []
baseline_paths = []


for path in os.listdir(case_folder):
    if path == 'compensation' or path == 'overview':
        continue
    # for i in range(1, 4):
    inpaint_paths.append(f'{RH_folder}/{path}/inpaint.png')
    baseline_paths.append(f'{case_folder}/{path}/inpaint.png')

baseline_score, inpaint_score = get_score(baseline_paths, inpaint_paths)

print(f'RH score: {inpaint_score}')
print(f'KK score: {baseline_score}')
