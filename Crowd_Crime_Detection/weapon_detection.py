import torch
from ultralytics import YOLO
import numpy as np
import cv2
from collections import deque
import threading
import queue
import time


class WeaponDetector:
    def __init__(self):
        # Load a pre-trained YOLOv8 model for weapon detection
        self.model = YOLO("runs/detect/train2/weights/best.pt")

        # Use a faster tracker - KCF instead of CSRT
        try:
            self.tracker = cv2.legacy.TrackerKCF_create()
            self.tracker_type = "legacy"
        except AttributeError:
            try:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker_type = "old"
            except AttributeError:
                self.tracker = None
                self.tracker_type = "none"
                print("Warning: KCF tracker not available. Using simple tracking.")

        # Parameters for tracking
        self.tracking_active = False
        self.tracking_box = None

        # IMPORTANT: Lowered back to original threshold to ensure weapons are detected
        self.confidence_threshold = 0.45
        self.last_detection = None

        # Memory buffer for high frame rates
        self.detection_memory = deque(maxlen=5)
        self.memory_threshold = 2

        # Parameters for reinitialization
        self.frames_since_detection = 0
        self.max_frames_without_detection = 10

        # For simple tracking
        self.tracked_weapons = []

        # For motion detection to filter false positives - disabled by default now
        self.use_motion_filtering = False  # IMPORTANT: Disabled motion filtering
        self.prev_frame = None
        self.min_motion_area = 50  # Reduced threshold

        # Multi-threading components
        self.detection_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()
        self.detection_thread_active = True
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        # Detection frequency control - IMPORTANT: Reduced to ensure we don't miss detections
        self.detect_every_n_frames = 2  # Only skip one frame
        self.frame_count = 0

        # Non-Maximum Suppression parameters - IMPORTANT: More lenient
        self.nms_threshold = 0.5  # Increased from 0.4

        # Debug mode
        self.debug_mode = True

    def _create_tracker(self):
        """Create a new tracker instance based on available OpenCV version"""
        if self.tracker_type == "legacy":
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == "old":
            return cv2.TrackerKCF_create()
        else:
            return None

    def _detection_worker(self):
        """Background thread for running YOLO detection"""
        while self.detection_thread_active:
            try:
                # Get a frame from the queue with timeout
                frame = self.detection_queue.get(timeout=1.0)

                # IMPORTANT: Use original size for better detection
                results = self.model(frame)  # Use default size for better accuracy

                # Process results
                current_detections = []
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()

                        if conf > self.confidence_threshold:
                            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                            frame_height, frame_width = frame.shape[:2]

                            # Boundary checks
                            x1 = max(0, min(x1, frame_width - 1))
                            y1 = max(0, min(y1, frame_height - 1))
                            x2 = max(0, min(x2, frame_width - 1))
                            y2 = max(0, min(y2, frame_height - 1))

                            # IMPORTANT: Reduced minimum size requirement
                            if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) > 50:
                                class_name = self.model.names[cls_id]
                                current_detections.append({
                                    "class": class_name,
                                    "confidence": conf,
                                    "box": (x1, y1, x2, y2)
                                })

                # Debug print for troubleshooting
                if self.debug_mode and current_detections:
                    print(f"Detection thread found {len(current_detections)} weapons")

                # Put results in the output queue
                self.result_queue.put(current_detections)

                # Mark task as done
                self.detection_queue.task_done()
            except queue.Empty:
                # Queue timeout, just continue
                continue
            except Exception as e:
                print(f"Detection thread error: {e}")
                continue

    def _check_motion(self, frame, box):
        """Check if there's significant motion in the detection area"""
        # If motion filtering is disabled, always return True
        if not self.use_motion_filtering:
            return True

        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        x1, y1, x2, y2 = box
        # Ensure coordinates are within frame boundaries
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        if x2 <= x1 or y2 <= y1:
            return False

        try:
            prev_roi = self.prev_frame[y1:y2, x1:x2]
            curr_roi = current_gray[y1:y2, x1:x2]

            # Handle edge cases where ROI dimensions don't match
            if prev_roi.shape != curr_roi.shape or prev_roi.size == 0 or curr_roi.size == 0:
                return True

            diff = cv2.absdiff(prev_roi, curr_roi)
            _, thresholded = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)  # Lower threshold
            motion_pixels = cv2.countNonZero(thresholded)

            # Update for next iteration
            self.prev_frame = current_gray

            # Return true if significant motion detected
            return motion_pixels > self.min_motion_area
        except Exception as e:
            print(f"Motion check error: {e}")
            return True

    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove overlapping boxes"""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        # NMS implementation
        keep = []

        while len(detections) > 0:
            # Keep the highest confidence detection
            keep.append(detections[0])

            # Remove overlapping detections
            remaining = []
            for i in range(1, len(detections)):
                # Calculate IoU with the kept detection
                box1 = detections[0]["box"]
                box2 = detections[i]["box"]

                # Calculate intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union = box1_area + box2_area - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou <= iou_threshold:
                        # Keep this detection if IoU is below threshold
                        remaining.append(detections[i])
                else:
                    # No intersection, keep the detection
                    remaining.append(detections[i])

            # Update the list of detections
            detections = remaining

        return keep

    def detect(self, frame):
        """
        Perform weapon detection on a frame with consistent tracking
        and multi-threaded detection for better performance
        """
        weapons_detected = False
        weapon_info = []
        frame_height, frame_width = frame.shape[:2]

        # Increment frame counter
        self.frame_count += 1

        # IMPORTANT: Submit frames more often
        if self.frame_count % self.detect_every_n_frames == 0 or self.frames_since_detection >= 10:
            # Try to add to queue without blocking
            try:
                if self.detection_queue.empty():
                    self.detection_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Queue is full, skip this frame

        # Get detection results if available
        current_detections = []
        try:
            while not self.result_queue.empty():
                current_detections = self.result_queue.get_nowait()
                self.result_queue.task_done()
        except queue.Empty:
            pass

        # Apply NMS to remove overlapping detections
        if current_detections:
            current_detections = self._apply_nms(current_detections, self.nms_threshold)

            if self.debug_mode:
                print(f"After NMS: {len(current_detections)} detections")

        # Add the best detection to memory and update tracker when we have detections
        if current_detections:
            # Find best detection by confidence
            best_detection = max(current_detections, key=lambda x: x["confidence"])

            # Verify with motion detection (if enabled)
            if self._check_motion(frame, best_detection["box"]):
                self.detection_memory.append(best_detection)

                # Initialize or reinitialize tracker
                x1, y1, x2, y2 = best_detection["box"]
                width, height = x2 - x1, y2 - y1

                if self.tracker_type != "none":
                    # Create and initialize tracker
                    self.tracker = self._create_tracker()
                    self.tracker.init(frame, (x1, y1, width, height))
                    self.tracking_active = True
                else:
                    # For simple tracking
                    self.tracked_weapons = []
                    for det in current_detections:
                        det["ttl"] = 10  # Time to live counter
                        self.tracked_weapons.append(det)

                self.frames_since_detection = 0
                self.last_detection = best_detection

                # Draw detection
                weapons_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{best_detection['class']}: {best_detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                weapon_info.append(best_detection)

                if self.debug_mode:
                    print(f"Weapon detected and tracker initialized")
            else:
                # False positive from motion detection
                self.detection_memory.append(None)
                if self.debug_mode:
                    print(f"Motion filter rejected detection")
        else:
            # No detections
            self.detection_memory.append(None)

        # Count recent positive detections
        recent_positive_detections = sum(1 for d in self.detection_memory if d is not None)

        # Deactivate tracking if we haven't had enough positive detections recently
        if recent_positive_detections < self.memory_threshold:
            if self.tracker_type != "none":
                self.tracking_active = False
            else:
                # For simple tracking, decrease TTL of tracked weapons
                for tracked in self.tracked_weapons:
                    if "ttl" in tracked:
                        tracked["ttl"] -= 1

                # Remove expired tracked weapons
                self.tracked_weapons = [t for t in self.tracked_weapons if t.get("ttl", 0) > 0]

        # Update tracking if it's active
        if self.tracking_active and self.tracker_type != "none":
            success, box = False, None
            try:
                # Update with appropriate API
                if self.tracker_type == "legacy":
                    success, box = self.tracker.update(frame)
                elif self.tracker_type == "old":
                    success, box = self.tracker.update(frame)
            except Exception as e:
                print(f"Tracker update error: {e}")
                success = False

            if success:
                # Convert to integer coordinates
                x, y, w, h = [int(v) for v in box]
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Boundary checks
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(0, min(x2, frame_width - 1))
                y2 = max(0, min(y2, frame_height - 1))

                if x2 > x1 and y2 > y1:  # Valid box
                    # Verify with motion detection if enabled
                    if self._check_motion(frame, (x1, y1, x2, y2)):
                        weapons_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        if self.last_detection:
                            class_name = self.last_detection["class"]
                            conf = self.last_detection["confidence"]
                        else:
                            class_name = "weapon"
                            conf = 0.5

                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        weapon_info.append({
                            "class": class_name,
                            "confidence": conf,
                            "box": (x1, y1, x2, y2)
                        })
                    else:
                        # Motion detection determined this is likely a false positive
                        self.frames_since_detection += 1
                else:
                    # Invalid box
                    self.frames_since_detection += 1
            else:
                # Tracking failed
                self.frames_since_detection += 1
        elif self.tracker_type == "none" and self.tracked_weapons:
            # Draw simple tracking results
            for weapon in self.tracked_weapons:
                x1, y1, x2, y2 = weapon["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                label = f"{weapon['class']}: {weapon['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                weapon_info.append({
                    "class": weapon["class"],
                    "confidence": weapon["confidence"],
                    "box": weapon["box"]
                })

            weapons_detected = len(self.tracked_weapons) > 0
        else:
            # No active tracking
            self.frames_since_detection += 1

        # Add debug info to frame if enabled
        if self.debug_mode:
            cv2.putText(frame, f"Detections in memory: {recent_positive_detections}/{len(self.detection_memory)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Frames since detection: {self.frames_since_detection}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Tracking active: {self.tracking_active}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame, weapon_info, weapons_detected

    def __del__(self):
        """Clean up resources"""
        self.detection_thread_active = False
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)


# Test function
def test_weapon_detector():
    detector = WeaponDetector()
    cap = cv2.VideoCapture(0)  # For webcam

    # Performance metrics
    fps_history = deque(maxlen=30)
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame, weapon_info, weapons_detected = detector.detect(frame)

        # Calculate FPS
        frame_count += 1
        if frame_count % 5 == 0:  # Update FPS every 5 frames
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

            # Display FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if weapons_detected:
            cv2.putText(frame, "WEAPON DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Weapon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_weapon_detector()