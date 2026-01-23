import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Environment Setup ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# --- Configs ---
SAM2_CHECKPOINT = "sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
YOLO_MODEL = 'yolo11s.pt' 
SKIP_FRAMES = 3  # SAM2 runs every 3rd frame for better responsiveness

# Initialization
model = YOLO(YOLO_MODEL)
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Pre-generate colors and get class names
np.random.seed(42)
COLOR_PALETTE = np.random.randint(0, 255, (100, 3)).tolist()
CLASS_NAMES = model.names 

def draw_mask(image, mask, obj_id, alpha=0.5):
    color = COLOR_PALETTE[int(obj_id) % len(COLOR_PALETTE)]
    if torch.is_tensor(mask): mask = mask.cpu().numpy()
    if mask.ndim == 3: mask = mask[0]
    
    mask = (mask > 0).astype(np.uint8)
    mask_indices = mask > 0
    if np.any(mask_indices):
        roi = image[mask_indices]
        blended = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        image[mask_indices] = blended
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1)
    return image

cap = cv2.VideoCapture(0)
frame_count = 0
last_masks, last_ids, last_cls = [], [], []

# Before the loop, find the 'person' ID dynamically
person_idx = [k for k, v in model.names.items() if v == 'person'][0]
all_indices = list(model.names.keys())
target_indices = [i for i in all_indices if i != person_idx]

print("Starting Multi-Object Tracker... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. YOLO Tracking (All classes, no filter)
    #results = model.track(frame, persist=True, conf=0.05, verbose=False)
    results = model.track(frame, persist=True, classes=target_indices, verbose=False)
    render_img = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        obj_ids = results[0].boxes.id.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

        # 2. SAM2 Mask Generation (Periodic)
        if frame_count % SKIP_FRAMES == 0:
            predictor.set_image(frame)
            masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
            last_masks, last_ids, last_cls = masks, obj_ids, cls_ids
        
        # 3. Rendering
        for i, current_id in enumerate(obj_ids):
            # Find the cached mask and class label for this ID
            if current_id in last_ids:
                idx = np.where(last_ids == current_id)[0][0]
                label = CLASS_NAMES[int(last_cls[idx])]
                
                # Draw Mask
                if idx < len(last_masks):
                    render_img = draw_mask(render_img, last_masks[idx], current_id)
                
                # Draw Bounding Box and Label
                x1, y1, x2, y2 = map(int, boxes[i])
                color = COLOR_PALETTE[int(current_id) % 100]
                cv2.rectangle(render_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(render_img, f"{label} {int(current_id)}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Universal SAM2 + YOLO Tracker", render_img)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()