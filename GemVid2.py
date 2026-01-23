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

if os.path.exists(FRAME_DIR): rmtree(FRAME_DIR)
os.makedirs(FRAME_DIR, exist_ok=True)

predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=DEVICE)

# State Management
clicked_point = None
def get_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cap = cv2.VideoCapture(0)
cv2.namedWindow("SAM2 Video Tracker")
cv2.setMouseCallback("SAM2 Video Tracker", get_click)

tracking_mode = False
inference_state = None
frame_idx = 0

print("Controls: 's' to Select | 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Save frame for SAM2
    frame_path = os.path.join(FRAME_DIR, f"{frame_idx:05d}.jpg")
    cv2.imwrite(frame_path, frame)

    if not tracking_mode:
        cv2.putText(frame, "LIVE: Press 's' to Select Object", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SAM2 Video Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # --- SELECTION PHASE ---
            clicked_point = None # Reset previous clicks
            print("Paused. Click the object once, then press ANY KEY to confirm.")
            
            while True:
                temp_img = frame.copy()
                if clicked_point:
                    cv2.circle(temp_img, clicked_point, 5, (0, 0, 255), -1)
                    cv2.putText(temp_img, "Point set! Press any key to track.", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(temp_img, "CLICK THE OBJECT", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("SAM2 Video Tracker", temp_img)
                # This inner waitKey allows the UI to refresh and wait for the click
                if cv2.waitKey(1) != -1 and clicked_point is not None:
                    break
            
            # --- INITIALIZE SAM2 ---
            inference_state = predictor.init_state(video_path=FRAME_DIR)
            points = np.array([clicked_point], dtype=np.float32)
            labels = np.array([1], np.int32)
            
            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                points=points,
                labels=labels,
            )
            tracking_mode = True

    else:
        # --- TRACKING/PROPAGATION PHASE ---
        # We propagate only the current frame to keep it 'live'
        # start_frame_idx and max_frame_num_to_track limits the propagation range
        _, out_obj_ids, out_mask_logits = next(predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=frame_idx, 
            max_frame_num_to_track=1 # Only track this new frame
        ))

        mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
        if mask.ndim == 3: mask = mask[0]
        
        # Overlay for visualization
        mask_indices = mask > 0
        if np.any(mask_indices):
            frame[mask_indices] = (frame[mask_indices] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

        cv2.putText(frame, "TRACKING (SAM2 Video Mode)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("SAM2 Video Tracker", frame)

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
rmtree(FRAME_DIR)