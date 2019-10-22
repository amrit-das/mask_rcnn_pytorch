import torchvision
import torch
from torchvision import transforms as T
from PIL import Image
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
model.cuda()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

'''colors = [ 
    (33, 71, 130), (35, 73, 132), (37, 75, 134), (39, 77, 136), (41, 79, 138), 
    (43, 81, 140), (45, 83, 142), (47, 85, 144), (49, 87, 146), (51, 89, 148), 
    (53, 91, 150), (55, 93, 152), (57, 95, 154), (59, 97, 156), (61, 99, 158), 
    (63, 101, 160), (65, 103, 162), (67, 105, 164), (69, 107, 166), (71, 109, 168), 
    (73, 111, 170), (75, 113, 172), (77, 115, 174), (79, 117, 176), (81, 119, 178), 
    (83, 121, 180), (85, 123, 182), (87, 125, 184), (89, 127, 186), (91, 129, 188), 
    (93, 131, 190), (95, 133, 192), (97, 135, 194), (99, 137, 196), (101, 139, 198), 
    (103, 141, 200), (105, 143, 202), (107, 145, 204), (109, 147, 206), (111, 149, 208), 
    (113, 151, 210), (115, 153, 212), (117, 155, 214), (119, 157, 216), (121, 159, 218), 
    (123, 161, 220), (125, 163, 222), (127, 165, 224), (129, 167, 226), (131, 169, 228), 
    (133, 171, 230), (135, 173, 232), (137, 175, 234), (139, 177, 236), (141, 179, 238), 
    (143, 181, 240), (145, 183, 242), (147, 185, 244), (149, 187, 246), (151, 189, 248), 
    (153, 191, 250), (155, 193, 252), (157, 195, 254), (159, 197, 0), (161, 199, 2), (163, 201, 4), 
    (165, 203, 6), (167, 205, 8), (169, 207, 10), (171, 209, 12), (173, 211, 14), (175, 213, 16), 
    (177, 215, 18), (179, 217, 20), (181, 219, 22), (183, 221, 24), (185, 223, 26), (187, 225, 28), 
    (189, 227, 30), (191, 229, 32), (193, 231, 34), (195, 233, 36), (197, 235, 38), (199, 237, 40), (201, 239, 42), 
    (203, 241, 44), (205, 243, 46), (207, 245, 48), (209, 247, 50), (211, 249, 52), (213, 251, 54)
]'''

def get_prediction(img_path, threshold):
    img = Image.fromarray(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.cuda()
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    #pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    #pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, "pass", pred_class

def colour_masks(image,i):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190],[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    r[image == 1], g[image == 1], b[image == 1] = colours[i]
    coloured_mask = np.stack([r, g, b],axis=2)
    print(type(coloured_mask))
    return coloured_mask


import cv2
cap = cv2.VideoCapture(0)
num_frames = 0
start_time = datetime.now()

while True:
    
    thresh = 0.7
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks, boxes, pred_cls = get_prediction(img, thresh)
    for i in range(len(masks)):
        rgb_mask = colour_masks(masks[i], i)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.7, 0)


        #cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        #cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    #num_frames = +1
    #elp_time = (datetime.now() - start_time).total_seconds()
    #fps = num_frames/elp_time

    #cv2.putText(img,str(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0))

    cv2.imshow("out",img)
    if cv2.waitKey(30) == 27:
        break

cap.release()    
cv2.destroyAllWindows()
    