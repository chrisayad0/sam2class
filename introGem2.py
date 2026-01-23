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

# --- Global Configs ---
SAM2_CHECKPOINT = "sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
YOLO_MODEL = 'yolo11n.pt'
SKIP_FRAMES = 4  # Run SAM2 every 4th frame

# Pre-generate stable colors
np.random.seed(42)
COLOR_PALETTE = np.random.randint(0, 255, (100, 3)).tolist()

# --- Initialization ---
model = YOLO(YOLO_MODEL)
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def draw_mask(image, mask, obj_id, alpha=0.5):
    color = COLOR_PALETTE[int(obj_id) % len(COLOR_PALETTE)]
    
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
        
    mask = (mask > 0).astype(np.uint8)
    
    # Fast alpha blending
    mask_indices = mask > 0
    if np.any(mask_indices):
        roi = image[mask_indices]
        blended = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        image[mask_indices] = blended

        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1)
    return image

cap = cv2.VideoCapture(0)
frame_count = 0
last_masks = []
last_ids = []

print("Starting stream... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO Tracking (Run every frame for smooth boxes)
    results = model.track(frame, persist=True, conf=0.25, verbose=False, classes=(0,9))
    render_img = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        obj_ids = results[0].boxes.id.cpu().numpy()

        # 2. Heavy SAM2 Logic (Only every N frames)
        if frame_count % SKIP_FRAMES == 0:
            predictor.set_image(frame)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )
            last_masks = masks
            last_ids = obj_ids
        
        # 3. Rendering logic
        # We only render masks for IDs that were present in the last SAM2 sweep
        for i, current_id in enumerate(obj_ids):
            # Check if this ID has a cached mask
            if current_id in last_ids:
                idx = np.where(last_ids == current_id)[0][0]
                # Ensure index doesn't exceed mask array size
                if idx < len(last_masks):
                    render_img = draw_mask(render_img, last_masks[idx], current_id)
            
            # Always draw the fresh YOLO box
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(render_img, (x1, y1), (x2, y2), COLOR_PALETTE[int(current_id)%100], 2)
            cv2.putText(render_img, f"ID: {int(current_id)}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("SAM2 + YOLO11 Tracking", render_img)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()