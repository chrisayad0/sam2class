import os
import cv2
import torch
import numpy as np
import easyocr
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Environment Setup ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Global Configs ---
SAM2_CHECKPOINT = "sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
YOLO_MODEL = 'yolo11n.pt'
SKIP_FRAMES = 4 
# COCO Classes: 73 is 'book'. 
# Note: COCO doesn't have 'paper'. Using 73 (book). 
# Adding 63 (laptop) or 28 (stop sign) is sometimes used for rectangular paper proxies.
TARGET_CLASSES = [73] 

# --- Initialization ---
model = YOLO(YOLO_MODEL)
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)
reader = easyocr.Reader(['en'], gpu=(device.type == 'cuda'))

# Accumulator for final print
extracted_text_registry = set()

def draw_mask(image, mask, obj_id, alpha=0.5):
    color = [0, 255, 0] # Standardize to green for clarity
    if torch.is_tensor(mask): mask = mask.cpu().numpy()
    if mask.ndim == 3: mask = mask[0]
    mask = (mask > 0).astype(np.uint8)
    mask_indices = mask > 0
    if np.any(mask_indices):
        roi = image[mask_indices]
        blended = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        image[mask_indices] = blended
    return image

cap = cv2.VideoCapture(0)
frame_count = 0
last_masks = []
last_ids = []

print("Scanning for books/paper... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. YOLO Tracking (Filtered for Books/Paper proxies)
    results = model.track(frame, persist=True, conf=0.3, verbose=False, classes=TARGET_CLASSES)
    render_img = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        obj_ids = results[0].boxes.id.cpu().numpy()

        # 2. SAM2 & OCR Logic (Every N frames to save compute)
        if frame_count % SKIP_FRAMES == 0:
            predictor.set_image(frame)
            masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
            last_masks = masks
            last_ids = obj_ids

            # OCR Pass on detected regions
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                # Crop with slight padding for OCR context
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    text_results = reader.readtext(crop, detail=0)
                    for text in text_results:
                        if len(text.strip()) > 2: # Filter noise
                            extracted_text_registry.add(text.strip())
        
        # 3. Rendering
        for i, current_id in enumerate(obj_ids):
            if current_id in last_ids:
                idx = np.where(last_ids == current_id)[0][0]
                if idx < len(last_masks):
                    render_img = draw_mask(render_img, last_masks[idx], current_id)
            
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(render_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(render_img, "READING...", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Book/Paper Text Detector", render_img)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- Final Console Output ---
print("\n" + "="*30)
print("EXTRACTED TEXT SUMMARY:")
print("="*30)
for item in sorted(extracted_text_registry):
    print(f"- {item}")
print("="*30)