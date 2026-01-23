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

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# --- Configs ---
SAM2_CHECKPOINT = "sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
YOLO_MODEL = 'yolo11s.pt'  # Small model for better recall
SKIP_FRAMES = 5

# Classes: 73 (book), 63 (laptop), 65 (remote), 67 (cell phone)
# Using multiple classes because thin books/paper are often misclassified
#TARGET_CLASSES = [73, 63, 65, 67] 
TARGET_CLASSES = [73, 63, 65, 67, 62, 66, 24, 26]

# Initialize OCR & Sets
reader = easyocr.Reader(['en'], gpu=(device.type == "cuda"))
found_texts = set()

# Pre-generate stable colors
np.random.seed(42)
COLOR_PALETTE = np.random.randint(0, 255, (100, 3)).tolist()

# --- Initialization ---
model = YOLO(YOLO_MODEL)
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

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
last_masks, last_ids = [], []

print("Stream started. Scanning for books/text...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. High-Sensitivity Tracking
    results = model.track(
        frame, 
        persist=True, 
        conf=0.01,      # Lowered threshold to find more books
        imgsz=1024,     # Increased resolution for tiny text
        verbose=False, 
        classes=TARGET_CLASSES
    )
    render_img = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        obj_ids = results[0].boxes.id.cpu().numpy()

        # 2. SAM2 & OCR Pass
        if frame_count % SKIP_FRAMES == 0:
            predictor.set_image(frame)
            masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
            last_masks, last_ids = masks, obj_ids

            # Crop and scan each detected box for text
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                # Crop with slight padding
                crop = frame[max(0, y1-5):min(frame.shape[0], y2+5), 
                             max(0, x1-5):min(frame.shape[1], x2+5)]
                
                if crop.size > 0:
                    # easyocr handles vertical/tilted spine text
                    ocr_results = reader.readtext(crop)
                    for (_, text, prob) in ocr_results:
                        if prob > 0.4 and len(text.strip()) > 3:
                            txt = text.strip()
                            if txt not in found_texts:
                                print(f"[FOUND]: {txt}")
                                found_texts.add(txt)
        
        # 3. Rendering Logic
        for i, current_id in enumerate(obj_ids):
            if current_id in last_ids:
                idx = np.where(last_ids == current_id)[0][0]
                if idx < len(last_masks):
                    render_img = draw_mask(render_img, last_masks[idx], current_id)
            
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(render_img, (x1, y1), (x2, y2), COLOR_PALETTE[int(current_id)%100], 2)
            cv2.putText(render_img, f"ID {int(current_id)}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow("Book/Paper Detection", render_img)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()



# # --- Updated Global Configs ---
# YOLO_MODEL = 'yolo11s.pt' # Switched to 'Small' from 'Nano' for better accuracy
# SKIP_FRAMES = 5
# # Classes: 73 (book), 63 (laptop - often paper), 66 (keyboard), 67 (cell phone)
# # We use these because paper is often misclassified as flat electronics
# TARGET_CLASSES = [73, 63, 66, 67] 

# # ... (rest of initialization remains same) ...

# while True:
#     ret, frame = cap.read()
#     if not ret: break

#     # 1. Improved YOLO Tracking
#     # Lower conf to 0.15 to find more objects; imgsz 640 for better detail
#     results = model.track(
#         frame, 
#         persist=True, 
#         conf=0.15,      # Lowered threshold
#         iou=0.5,       # Adjust overlap sensitivity
#         imgsz=640,     # Explicitly set higher resolution
#         verbose=False, 
#         classes=TARGET_CLASSES
#     )
    
#     render_img = frame.copy()

#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy
#         obj_ids = results[0].boxes.id.cpu().numpy()
#         class_ids = results[0].boxes.cls.cpu().numpy()

#         # 2. OCR Logic
#         if frame_count % SKIP_FRAMES == 0:
#             predictor.set_image(frame)
#             masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
#             last_masks, last_ids = masks, obj_ids

#             for i, box in enumerate(boxes):
#                 x1, y1, x2, y2 = map(int, box)
#                 # Expand crop slightly for better OCR context
#                 pad = 5
#                 crop = frame[max(0, y1-pad):min(frame.shape[0], y2+pad), 
#                              max(0, x1-pad):min(frame.shape[1], x2+pad)]
                
#                 if crop.size > 0:
#                     # EasyOCR is great for rotated spine text
#                     ocr_results = reader.readtext(crop, contrast_ths=0.2, brightness_ths=0.2)
#                     for (_, text, prob) in ocr_results:
#                         if prob > 0.3 and len(text.strip()) > 2:
#                             found_texts.add(text.strip())
#                             print(f"[Text Found]: {text.strip()} (Conf: {prob:.2f})")

#         # ... (Rendering remains same) ...