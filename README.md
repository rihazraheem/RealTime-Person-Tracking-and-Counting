A real-time computer vision application for **person tracking and counting** from a live webcam feed.

This project combines the power of **YOLOv8** for high-accuracy person detection with the **DeepSORT** algorithm for robust multi-object tracking, enabling the counting of:
1.  **Total Unique** individuals seen.
2.  **Currently Present** individuals.
3.  **Exited** individuals (Total - Present).

## ✨ Key Features

* **Real-Time Processing:** Tracks and counts people directly from a webcam.
* **Robust Tracking:** Uses DeepSORT to assign and maintain unique IDs for each person, even through occlusions.
* **Counting Logic:** Calculates total unique entries, current occupancy, and people who have exited the frame.
* **Video Output:** Optionally saves the tracked video output to a file (`webcam_tracked_counts.mp4`).

## 🛠️ Prerequisites

Before running the script, ensure you have Python installed (version 3.8+ recommended) and a working webcam.
