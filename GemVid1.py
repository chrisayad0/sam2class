import os
import cv2
import torch
import numpy as np
from shutil import rmtree
from sam2.build_sam import build_sam2_video_predictor

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT = "sam2.1_hiera_small.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
FRAME_DIR = "./temp_frames"

# Clean/Create frame directory
if os.path.exists(FRAME_DIR): rmtree(FRAME_DIR)
os.makedirs(FRAME_DIR, exist_ok=True)

predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=DEVICE)

# Global variables for mouse callback
clicked_point = None
def get_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cap = cv2.VideoCapture(0)
cv2.namedWindow("SAM2 Video Tracker")
cv2.setMouseCallback("SAM2 Video Tracker", get_click)

print("1. Press 's' to freeze and select an object.")
print("2. Click the object, then press any key to start tracking.")
print("3. Press 'q' to quit.")

tracking_mode = False
inference_state = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # Save frame for SAM2 (Required for Video Predictor state)
    frame_path = os.path.join(FRAME_DIR, f"{frame_idx:05d}.jpg")
    cv2.imwrite(frame_path, frame)

    if not tracking_mode:
        display_frame = frame.copy()
        cv2.putText(display_frame, "LIVE: Press 's' to Select", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SAM2 Video Tracker", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # --- SELECTION PHASE ---
            print("Frozen. Click the object.")
            while clicked_point is None:
                cv2.imshow("SAM2 Video Tracker", frame)
                cv2.waitKey(1)
            
            # Initialize state with frames captured so far
            inference_state = predictor.init_state(video_path=FRAME_DIR)
            
            # Add click prompt
            points = np.array([clicked_point], dtype=np.float32)
            labels = np.array([1], np.int32) # 1 = foreground
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                points=points,
                labels=labels,
            )
            tracking_mode = True
            print("Tracking started...")

    else:
        # --- TRACKING PHASE ---
        # Propagate from current frame index
        # Note: propagate_in_video is a generator; we take the next step
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx):
            # Process only the current frame
            if out_frame_idx == frame_idx:
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                if mask.ndim == 3: mask = mask[0]
                
                # Overlay Mask
                overlay = frame.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                
                # Blended look
                mask_indices = mask > 0
                overlay[mask_indices] = (overlay[mask_indices] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                
                cv2.imshow("SAM2 Video Tracker", overlay)
                break # Move to next capture frame

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
rmtree(FRAME_DIR)