import numpy as np
import os
from PIL import Image
import cv2

img = Image.open('dataset/near-infrared/00/train_pose/02_l/frame_4292_l.png')
arr = np.array(img)
cv2.imwrite('Arr.jpg', arr)
img = cv2.GaussianBlur(arr, (5, 5), 0) 
cv2.imwrite('Gausian.jpg', img)
thresholded = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('Thresholded.jpg', img)
    
