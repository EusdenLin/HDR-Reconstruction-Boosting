from brisque import BRISQUE
import cv2


# path = '../EV-3_GT.png'
# path = '../EV-3_CEVR.png'
path = '../0909_gammaTMO/results_0.05/-3.png'

img = cv2.imread(path)
obj = BRISQUE(url=False)
score = obj.score(img)
print(score)