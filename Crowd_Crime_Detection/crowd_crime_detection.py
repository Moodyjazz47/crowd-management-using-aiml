import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse

# Import our detection modules
from weapon_detection import WeaponDetector
from fight_detection import FightDetector
from anomaly_detection import AnomalyDetector


# Defining CSRNet model

class CSRNet(torch.nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.backend = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


class CrowdCrimeDetectionSystem:
    def __init__(self, video_source="samplevid2.mp4", use_gpu=True):
        """
        Initialize the integrated crowd and crime detection system
        """
        self.video_source = video_source
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        print(f"Initializing system with {'GPU' if self.use_gpu else 'CPU'} support...")

        # Initialize crowd management components
        self.yolo_model = YOLO("yolov8n.pt")
        self.csrnet = CSRNet()
        if self.use_gpu:
            self.csrnet = self.csrnet.cuda()

        try:
            self.csrnet.load_state_dict(torch.load("csrnet_crowd.pth",
                                                   map_location=self.device))
            print("CSRNet model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CSRNet model: {e}")
            print("System will continue with YOLO-only crowd detection")

        self.csrnet.eval()

        # Initialize crime detection components
        print("Loading crime detection modules...")
        self.weapon_detector = WeaponDetector()
        self.fight_detector = FightDetector()
        self.anomaly_detector = AnomalyDetector()

        print("All components initialized successfully!")

        # Display settings
        self.display_mode = "all"  # Options: "all", "crowd", "weapons", "fights", "anomalies"
        self.frame_count = 0

        # Parameters for people counting
        self.people_count_history = []  # Store recent counts for smoothing
        self.history_size = 5  # Number of frames to average
        self.max_reasonable_density = 5.0  # Max density per 1000 pixels
        self.csrnet_adjustment_factor = 0.8  # Scale factor for CSRNet estimation
        self.use_yolo_validation = True  # Use YOLO counts to validate CSRNet

    def process_frame(self, frame):
        """
        Process a single frame with all detection components
        """
        # Create a working copy to prevent sequential modifications from interfering
        display_frame = frame.copy()

        # Track processing time
        start_time = time.time()

        # PART 1: CROWD MANAGEMENT (with fixes for people counting)
        # YOLO People Counting
        results = self.yolo_model(frame)

        # Extract actual person detections
        people_boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()

                # Only count person class (0) with good confidence
                if cls_id == 0 and conf > 0.5:
                    x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                    people_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        people_count_yolo = len(people_boxes)

        # CSRNet Density Estimation (with improved handling)
        try:
            # Calculate frame area (excluding letterboxing if present)
            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height / 1000  # Area in kilo-pixels

            # Process through CSRNet
            input_frame = cv2.resize(frame, (256, 256))
            input_tensor = torch.tensor(input_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

            if self.use_gpu:
                input_tensor = input_tensor.cuda()

            with torch.no_grad():
                density_map = self.csrnet(input_tensor).cpu().squeeze().numpy()

            # Scale density map to frame size for visualization
            density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))

            # Calculate count based on density map sum with adjustment
            raw_density_sum = np.sum(density_map)

            # Apply scaling based on frame area and our adjustment factor
            people_count_csrnet = int(np.round(raw_density_sum * self.csrnet_adjustment_factor))

            # Density sanity check - limit maximum reasonable density
            max_reasonable_count = int(frame_area * self.max_reasonable_density)
            if people_count_csrnet > max_reasonable_count:
                people_count_csrnet = max_reasonable_count

            # If YOLO validation is enabled, use YOLO count as upper bound
            # when CSRNet is significantly higher (>3x) than YOLO count
            if self.use_yolo_validation:
                # If CSRNet count is much higher than YOLO, prefer YOLO
                if people_count_yolo > 0 and people_count_csrnet > 3 * people_count_yolo:
                    people_count_csrnet = people_count_yolo

            # Normalize density map for visualization
            density_viz = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5) * 255
            density_viz = density_viz.astype(np.uint8)
            density_map_colored = cv2.applyColorMap(density_viz, cv2.COLORMAP_JET)

            # Prepare combined visualization for crowd
            crowd_viz = cv2.addWeighted(frame, 0.6, density_map_colored, 0.4, 0)

            # Final People Count: Use a more intelligent merging of YOLO & CSRNet
            # Balance reliability of YOLO for low counts with CSRNet for dense crowds
            if people_count_yolo <= 10:  # For sparse crowds, YOLO is more reliable
                people_count_final = people_count_yolo
            else:  # For dense crowds, use a weighted average
                people_count_final = int(0.7 * people_count_yolo + 0.3 * people_count_csrnet)

        except Exception as e:
            print(f"Error in CSRNet processing: {e}")
            people_count_final = people_count_yolo
            crowd_viz = frame.copy()

        # Apply temporal smoothing to reduce count fluctuations
        self.people_count_history.append(people_count_final)
        if len(self.people_count_history) > self.history_size:
            self.people_count_history.pop(0)

        smoothed_count = int(round(sum(self.people_count_history) / len(self.people_count_history)))

        # PART 2: CRIME DETECTION
        # Weapon detection
        _, weapon_info, weapons_detected = self.weapon_detector.detect(frame)

        # Fight detection using pose estimation
        fight_frame, fight_detected, _ = self.fight_detector.detect(frame)

        # Pass frame count to anomaly detector for temporal context
        self.frame_count += 1  # Make sure to increment frame counter
        self.anomaly_detector.set_frame_count(self.frame_count)

        # Anomaly detection - now focused on theft and suspicious behaviors
        anomaly_frame, theft_detected, suspicious_regions = self.anomaly_detector.detect(frame)

        # Combine all detections on the display frame
        if self.display_mode == "all" or self.display_mode == "crowd":
            # Add crowd information to display frame
            cv2.putText(display_frame, f"People Count: {smoothed_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add density map overlay if in crowd mode
            if self.display_mode == "crowd":
                display_frame = cv2.addWeighted(display_frame, 0.6, density_map_colored, 0.4, 0)

        # Add weapon detection info
        if self.display_mode == "all" or self.display_mode == "weapons":
            if weapons_detected:
                cv2.putText(display_frame, "WEAPON DETECTED!",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw weapon boxes
                for weapon in weapon_info:
                    x1, y1, x2, y2 = weapon["box"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"{weapon['class']}: {weapon['confidence']:.2f}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Add fight detection info
        if self.display_mode == "all" or self.display_mode == "fights":
            if fight_detected:
                cv2.putText(display_frame, "FIGHT DETECTED!",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # If in fight mode only, show the pose visualization
            if self.display_mode == "fights":
                display_frame = fight_frame

        # Add theft/anomaly detection info (updated for new theft detection)
        if self.display_mode == "all" or self.display_mode == "anomalies":
            if theft_detected:
                cv2.putText(display_frame, "POTENTIAL THEFT DETECTED!",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw theft detection regions
                for x, y, w, h in suspicious_regions:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # If in anomaly mode only, show the theft visualization
            if self.display_mode == "anomalies":
                display_frame = anomaly_frame

        # Calculate and display FPS
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}",
                    (display_frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Return the integrated results
        detection_results = {
            "people_count": smoothed_count,
            "people_count_yolo": people_count_yolo,
            "people_count_csrnet": people_count_csrnet if 'people_count_csrnet' in locals() else 0,
            "weapons_detected": weapons_detected,
            "weapon_info": weapon_info,
            "fight_detected": fight_detected,
            "theft_detected": theft_detected,  # Updated from "anomaly_detected"
            "suspicious_regions": suspicious_regions,
            "fps": fps
        }

        return display_frame, detection_results

    def run(self):
        """
        Main execution loop for video processing
        """
        # Open video source
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{self.video_source}'")
            return

        # Get video properties for potential recording
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Video dimensions: {width}x{height}, FPS: {fps}")
        print("Starting processing pipeline...")
        print("Press 'q' to quit, 'm' to change display mode")

        # Initialize display window
        cv2.namedWindow("Crowd and Crime Detection", cv2.WINDOW_NORMAL)

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached")
                break

            # Process the frame with all detection components
            processed_frame, results = self.process_frame(frame)

            # Display the results
            cv2.imshow("Crowd and Crime Detection", processed_frame)

            # Display current mode and detailed counts (for debugging)
            mode_text = f"Mode: {self.display_mode.upper()}"
            cv2.putText(processed_frame, mode_text,
                        (processed_frame.shape[1] - 200, processed_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Cycle through display modes
                modes = ["all", "crowd", "weapons", "fights", "anomalies"]
                current_index = modes.index(self.display_mode)
                next_index = (current_index + 1) % len(modes)
                self.display_mode = modes[next_index]
                print(f"Switched to {self.display_mode.upper()} display mode")
            elif key == ord('d'):
                # Add a debug mode to display raw counts
                yolo_count = results["people_count_yolo"]
                csrnet_count = results["people_count_csrnet"]
                final_count = results["people_count"]
                print(f"Debug - YOLO: {yolo_count}, CSRNet: {csrnet_count}, Final: {final_count}")

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Processing complete")


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Crowd Management and Crime Detection System")
    parser.add_argument("--video", type=str, default="samplevid2.mp4",
                        help="Path to input video file (default: samplevid2.mp4)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "crowd", "weapons", "fights", "anomalies"],
                        help="Initial display mode (default: all)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    args = parser.parse_args()

    # Create and run the system
    system = CrowdCrimeDetectionSystem(
        video_source=args.video,
        use_gpu=not args.cpu
    )
    system.display_mode = args.mode
    system.run()


if __name__ == "__main__":
    main()