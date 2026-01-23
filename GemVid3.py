import os
import cv2
import torch
import numpy as np
from shutil import rmtree
from sam2.build_sam import build_sam2_video_predictor

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT = "sam2.1_hiera_small.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
FRAME_DIR = "./live_frames"
WINDOW_SIZE = 30  # Number of past frames to keep for temporal memory

# --- Environment Setup ---
if os.path.exists(FRAME_DIR): rmtree(FRAME_DIR)
os.makedirs(FRAME_DIR, exist_ok=True)

predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=DEVICE)

# --- Global State ---
clicked_point = None
tracking_mode = False
inference_state = None
frame_idx = 0 
active_frame_names = []

def get_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cap = cv2.VideoCapture(0)
cv2.namedWindow("SAM2 Sliding Window Tracker")
cv2.setMouseCallback("SAM2 Sliding Window Tracker", get_click)

print("Controls: 's' to Select | 'r' to Reset | 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Manage Sliding Window Files
    frame_name = f"{frame_idx:05d}.jpg"
    frame_path = os.path.join(FRAME_DIR, frame_name)
    cv2.imwrite(frame_path, frame)
    active_frame_names.append(frame_name)

    # Cleanup old frames from disk to prevent storage bloat
    if len(active_frame_names) > WINDOW_SIZE:
        oldest_frame = active_frame_names.pop(0)
        old_path = os.path.join(FRAME_DIR, oldest_frame)
        if os.path.exists(old_path): os.remove(old_path)

    if not tracking_mode:
        # --- IDLE/SELECTION MODE ---
        display_msg = "LIVE: Press 's' to Select"
        cv2.putText(frame, display_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SAM2 Sliding Window Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            clicked_point = None
            while True:
                temp_img = frame.copy()
                msg = "CLICK OBJECT, then press ANY KEY" if not clicked_point else "POINT SET! Press ANY KEY"
                color = (255, 255, 255) if not clicked_point else (0, 0, 255)
                
                if clicked_point:
                    cv2.circle(temp_img, clicked_point, 5, (0, 0, 255), -1)
                
                cv2.putText(temp_img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("SAM2 Sliding Window Tracker", temp_img)
                
                if cv2.waitKey(1) != -1 and clicked_point:
                    break
            
            # Initialize State on current frame window
            if inference_state: predictor.reset_state(inference_state)
            inference_state = predictor.init_state(video_path=FRAME_DIR)
            
            # Map global frame_idx to the local window index (usually the last frame)
            local_idx = len(active_frame_names) - 1
            predictor.add_new_points(inference_state, frame_idx=local_idx, obj_id=1, 
                                     points=np.array([clicked_point], dtype=np.float32),
                                     labels=np.array([1], np.int32))
            tracking_mode = True

    else:
        # --- TRACKING MODE ---
        # Note: In sliding window, we always track the 'latest' frame in the state
        current_local_idx = len(active_frame_names) - 1
        
        try:
            # We propagate from the last known good frame to the current frame
            # In live mode, we just want the result for the most recent frame
            gen = predictor.propagate_in_video(inference_state, start_frame_idx=current_local_idx, max_frame_num_to_track=1)
            
            for out_frame_idx, out_obj_ids, out_mask_logits in gen:
                if len(out_obj_ids) > 0:
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                    if mask.ndim == 3: mask = mask[0]
                    
                    # Visualization
                    overlay = frame.copy()
                    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                    frame = overlay

        except Exception as e:
            print(f"Tracking lost or error: {e}")
            tracking_mode = False

        cv2.putText(frame, "TRACKING: Press 'r' to Reset", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("SAM2 Sliding Window Tracker", frame)

    # --- Global Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('r'):
        tracking_mode = False
        if inference_state: predictor.reset_state(inference_state)

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
if os.path.exists(FRAME_DIR): rmtree(FRAME_DIR)