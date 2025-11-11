from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os

# ---- Models ----
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7,
                   max_cosine_distance=0.2, nn_budget=100)

# ---- Capture (webcam) ----
cap = cv2.VideoCapture(0)  # use 0 for webcam, or path to video
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index 0).")

# Optionally force a resolution (helps stabilise width/height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps    = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 20.0

# ---- Output writer (ensure dir exists) ----
out_path = 'webcam_tracked_counts.mp4'
os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))

# ---- Counting state ----
unique_ids = set()    # all unique confirmed track IDs ever seen
prev_ids = set()      # IDs present in previous frame
current_ids = set()   # IDs present in current frame
entered_count = 0     # cumulative number of entries
exited_count = 0      # cumulative number of exits
exited_ids = set()    # optional: IDs that have exited at least once

frame_idx = 0
start_time = time.time()
print("Starting webcam tracking. Press 'q' to quit.")

# ---- Main loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed; exiting.")
        break

    frame_idx += 1
    current_ids.clear()

    # Detect persons every frame (remove/do_detect logic if you want)
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

    # Draw tracks and build current_ids
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        unique_ids.add(tid)
        current_ids.add(tid)

        # get bbox and draw
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {tid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---- compute transitions ----
    new_entries = current_ids - prev_ids    # appeared this frame
    new_exits   = prev_ids - current_ids    # disappeared this frame

    entered_count += len(new_entries)
    exited_count += len(new_exits)
    exited_ids.update(new_exits)

    # update prev_ids for next frame
    prev_ids = current_ids.copy()

    # ---- overlay counts ----
    total_unique = len(unique_ids)
    currently_present = len(current_ids)

    cv2.putText(frame, f'Total unique: {total_unique}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.putText(frame, f'Present now: {currently_present}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f'Entered (cum): {entered_count}', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, f'Exited (cum): {exited_count}', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # show and save
    cv2.imshow("Webcam Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- cleanup ----
cap.release()
out.release()
cv2.destroyAllWindows()
elapsed = time.time() - start_time
print(f"Stopped. Processed {frame_idx} frames in {elapsed:.1f}s (avg {frame_idx/elapsed:.1f} FPS).")
print("Tracking complete! Video saved at:", out_path)
print(f"Total unique people detected: {total_unique}")
print(f"Currently present in last frame: {currently_present}")
print(f"Entered (cum): {entered_count}")
print(f"Exited (cum): {exited_count}")
