from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Environment and Device Setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# --- Initialization (Outside Loop) ---
model = YOLO('yolo11n.pt')
sam2_checkpoint = "sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def draw_mask(image, mask, random_color=False, borders=True):
    if random_color:
        color = np.random.randint(0, 255, (3,)).tolist()
    else:
        color = [255, 144, 30] 
        
    # Ensure mask is a 2D numpy array
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Handle SAM2 output shapes (C, H, W) or (H, W)
    if mask.ndim == 3:
        mask = mask[0]
        
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    
    # Resize mask if it doesn't match image resolution
    if (h, w) != (image.shape[0], image.shape[1]):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_overlay = image.copy()
    mask_overlay[mask > 0] = color
    
    alpha = 0.4
    cv2.addWeighted(mask_overlay, alpha, image, 1 - alpha, 0, image)

    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=1)
    return image

def draw_box(image, box):
    coords = [int(x) for x in box]
    cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
    return image

cap = cv2.VideoCapture(0)
cv2.namedWindow("main window")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Inference
    results = model(frame, conf=0.25, verbose=False, classes=(0,9))
    boxes_yolo = results[0].boxes.xyxy # (N, 4)

    render_img = frame.copy()

    if len(boxes_yolo) > 0:
        # SAM2 Prediction
        predictor.set_image(frame)
        # boxes_yolo is already (N, 4), which predictor.predict expects
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_yolo,
            multimask_output=False,
        )

        # Iterate through detections and their corresponding SAM masks
        # masks shape: (N, 1, H, W)
        for i in range(len(boxes_yolo)):
            render_img = draw_mask(render_img, masks[i], random_color=True)
            render_img = draw_box(render_img, boxes_yolo[i])

    cv2.imshow("main window", render_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()