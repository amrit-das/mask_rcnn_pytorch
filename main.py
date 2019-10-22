import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
import numpy as np 
import random 
from matplotlib import pyplot as plt

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
model.cuda()

def get_prediction(img, threshold):
    img = Image.fromarray(img)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.cuda()
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    
    masks = masks[:pred_t+1]
    return masks 

def color_masks(image,i):
    colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190],[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r,g,b = np.zeros_like(image).astype(np.uint8), np.zeros_like(image).astype(np.uint8), np.zeros_like(image).astype(np.uint8)

    r[image == 1], g[image == 1], b[image == 1] = colors[i] 
    colored_masks = np.stack([r,g,b],axis = 2)
    
    return colored_masks

import cv2
cap = cv2.VideoCapture(0)

while True:
    thresh = 0.9
    ret,frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = get_prediction(img, thresh)

    for i in range(len(masks)):
        rgb_mask = color_masks(masks[i],i)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.7, 0.7, 0)
    cv2.imshow("frame",frame)
    cv2.imshow("mask",rgb_mask)
    cv2.imshow("screen",img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

