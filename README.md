# Crowd Management and Crime Detection Using AI/ML

This project provides an integrated system for advanced crowd monitoring and automated crime detection in surveillance footage. It leverages deep learning and computer vision to deliver the following key capabilities:

## Features

- **Crowd Count Estimation**
  - Utilizes a deep learning-based density estimation model (CSRNet) to accurately estimate the number of people present in a scene, even in dense crowds.
  - Maintains a smoothed count history for stable and reliable reporting.

- **Weapon Detection**
  - Implements YOLO-based object detection to identify weapons (such as guns and knives) in real-time video feeds.
  - Draws bounding boxes around detected weapons to alert security personnel.

- **Anomaly and Theft Detection**
  - Uses a custom anomaly detection module to identify suspicious behavior, with a strong focus on theft scenarios.
  - Tracks individuals and items, analyzes movement patterns, and flags activities like item transfers, suspicious movement, and behavior (e.g., wearing a red hoodie—which is often used in theft datasets).
  - Highlights suspicious regions and individuals with bounding boxes and on-screen alerts.

- **Fight Detection Using Pose Analysis**
  - Integrates MediaPipe-based pose estimation to monitor body postures and identify fights.
  - Detects raised hands, close proximity, and fast limb movements as indicators of physical altercations.
  - Annotates ongoing fights directly on the video frame with pose landmarks and alerts.

## How It Works

1. **Frame Processing:**  
   Each video frame is processed by the main detection pipeline, which runs all detection components in parallel.

2. **People Counting:**  
   The system estimates crowd density using CSRNet, providing both instantaneous and smoothed crowd counts.

3. **Weapon Detection:**  
   The weapon detection module uses YOLO object detection to scan for weapons and overlays bounding boxes where threats are detected.

4. **Anomaly & Theft Detection:**  
   The anomaly detector processes the frame to look for suspicious behaviors, such as theft, by tracking people and items, analyzing trajectories, and applying behavioral heuristics.

5. **Fight Detection:**  
   The pose-based fight detector uses MediaPipe to extract body landmarks, monitor interactions, and recognize aggressive or violent behaviors through pose analysis.

6. **Visualization:**  
   All events (crowd counts, weapon detections, anomalies, fights) are visualized on the video stream with bounding boxes and alert messages for easy monitoring.

## Datasets Used

- **Crowd Counting:** Trained on publicly available crowd datasets suitable for density estimation.
- **Weapons and Anomalies:** Trained on custom or open datasets containing labeled weapons and theft scenarios.
- **Fight Detection:** Utilizes pose estimation data, likely enhanced with fight/non-fight video samples for supervised learning.

## Applications

- Public safety in crowded venues (stadiums, malls, festivals, public transport hubs)
- Automated surveillance for theft and violence prevention in retail and urban environments
- Support for law enforcement and security agencies through real-time alerts

## Getting Started

Clone the repository, install the required dependencies, and run the main detection script on your video feeds. The system is modular, allowing you to enable or disable specific detection components as needed.

---

**Note:**  
- The system is designed for research and development purposes. Real-world deployment requires further tuning, rigorous evaluation, and consideration of privacy and ethical guidelines.