from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os

# Load models
model = YOLO('yolov8n.pt')  # YOLOv8 person detection
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7,max_cosine_distance=0.2, nn_budget=100) # DeepSORT tracker

# Webcam capture
cap = cv2.VideoCapture(0)  #change 0 to a path to a video file, if you need to use a recorded video.
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index 0).")

# Get webcam frame size
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 20  # default 20 if webcam returns 0

# Optional: To save output video
out_path = 'webcam_tracked_counts.mp4'
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),fps, (width, height))

#Initialize counting 
unique_ids = set()    # all unique confirmed track IDs ever seen
current_ids = set()   # track IDs in current frame
exited_ids = set()    # track IDs that were previously present but now absent
frame_idx = 0
last_detect_frame = -999

print("Starting webcam tracking. Press 'q' to quit.")
start_time = time.time()

# Real-time processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed; exiting.")
        break

    frame_idx += 1  
  
    # Detect persons with YOLO
    results = model.predict(frame, conf=0.35, iou=0.45, classes=[0], imgsz=640, verbose=False)[0]

    # Prepare detections for DeepSORT
    dets = []
    if hasattr(results, "boxes") and len(results.boxes) > 0:
        for i in range(len(results.boxes)):
            xyxy = results.boxes.xyxy[i].cpu().numpy()
            conf = float(results.boxes.conf[i].cpu().numpy())
            if conf < 0.35:
              continue
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
              continue
            
            dets.append(([x1, y1, w, h], conf, "person"))

    # Update tracker
    tracks = tracker.update_tracks(dets, frame=frame)

    current_ids.clear()  # reset current frame IDs

    #Draw boxes and update IDs
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        unique_ids.add(track_id)
        current_ids.add(track_id)

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate counts
    total_unique = len(unique_ids)
    currently_present = len(current_ids)
    exited = total_unique - currently_present

    # Display counts on frame
    cv2.putText(frame, f'Total: {total_unique}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f'Present: {currently_present}', (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Exited: {exited}', (10,110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    #Show live frame
    cv2.imshow("Webcam Tracking", frame)
    out.write(frame)  # optional: save video

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
elapsed = time.time() - start_time
print(f"Stopped. Processed {frame_idx} frames in {elapsed:.1f}s (avg {frame_idx/elapsed:.1f} FPS).")
print("Tracking complete! Video saved at:", out_path)
print(f"Total unique people detected: {total_unique}")
print(f"Currently present in last frame: {currently_present}")
print(f"Exited people: {exited}")
