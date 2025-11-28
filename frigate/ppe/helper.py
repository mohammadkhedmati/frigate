import os
import cv2
import numpy as np
from ultralytics import YOLO

REID_HIST_BINS = 8
REID_CROP_SIZE = (64, 64)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_yolo11.pt")

def safe_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = YOLO(path)
    # try move to GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            model.to("cuda")
            print("[INFO] Using CUDA for inference")
    except Exception:
        pass
    return model

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    inter = interW * interH
    if inter == 0: return 0.0
    union = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter
    return inter / union if union>0 else 0.0

def center_point(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def crop_rgb(img, box):
    x1,y1,x2,y2 = box
    h,w = img.shape[:2]
    x1 = max(0,int(x1)); y1=max(0,int(y1))
    x2 = min(w-1,int(x2)); y2 = min(h-1,int(y2))
    if x2<=x1 or y2<=y1:
        return None
    crop = img[y1:y2, x1:x2]
    try:
        crop_small = cv2.resize(crop, REID_CROP_SIZE, interpolation=cv2.INTER_AREA)
        crop_rgb = cv2.cvtColor(crop_small, cv2.COLOR_BGR2RGB)
        return crop_rgb
    except Exception:
        return None

def color_hist_feature(img_rgb, bins=REID_HIST_BINS):
    try:
        h = []
        for i in range(3):
            ch = img_rgb[:,:,i]
            hist,_ = np.histogram(ch.flatten(), bins=bins, range=(0,255))
            h.append(hist)
        feat = np.concatenate(h).astype(np.float32)
        s = feat.sum()
        if s > 0:
            feat = feat / s
        return feat
    except Exception:
        return None

def cosine_sim(a,b):
    if a is None or b is None: return 0.0
    denom = (np.linalg.norm(a)*np.linalg.norm(b))
    if denom==0: return 0.0
    return float(np.dot(a,b)/denom)

def draw_box_cv(frame, box, text=None, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (tw, th0), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.rectangle(frame, (x1, y1 - th0 - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4), font, scale, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
    return frame
