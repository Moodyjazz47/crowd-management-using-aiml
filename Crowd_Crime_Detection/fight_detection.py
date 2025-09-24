import cv2
import numpy as np
import time
import mediapipe as mp
from scipy.spatial import distance


class PoseBasedFightDetector:
    """
    Pose-based fight detection using MediaPipe pose estimation
    This approach analyzes body postures and relative positions to detect fighting behavior
    """

    def __init__(self):
        print("Initializing MediaPipe pose detection...")
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create pose estimator with default parameters
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Thresholds for fight detection
        self.proximity_threshold = 0.15  # Normalized distance between people
        self.raised_hands_threshold = 0.8  # Height ratio for detecting raised hands
        self.limb_speed_threshold = 0.03  # Threshold for fast limb movement

        # Store previous frame's pose data for motion analysis
        self.prev_poses = []

        # Detection memory to reduce flickering
        self.detection_memory = []
        self.memory_length = 5
        self.detection_threshold = 0.6  # Percentage of positive detections needed

        # Tracking for pose stability
        self.frame_count = 0
        self.stabilization_frames = 5

    def detect(self, frame):
        """
        Detect fights using pose estimation
        Returns: annotated frame, fight detection flag, and dummy score (for compatibility)
        """
        self.frame_count += 1

        # Make a copy of the frame for visualization
        viz_frame = frame.copy()

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Convert the BGR image to RGB before processing
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # If no pose is detected
        if not results.pose_landmarks:
            return viz_frame, False, 0.0

        # Extract landmarks from the current pose
        current_landmarks = []
        if results.pose_landmarks:
            # Draw the pose landmarks on the image
            self.mp_drawing.draw_landmarks(
                viz_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

            # Convert normalized landmarks to pixel coordinates
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                landmarks.append((x, y, landmark.visibility))

            current_landmarks.append(landmarks)

        # First frame initialization or stabilization period
        if not self.prev_poses or self.frame_count < self.stabilization_frames:
            self.prev_poses = current_landmarks
            return viz_frame, False, 0.0

        # Fight detection logic
        fight_indicators = []

        # Process individual pose for fighting indicators
        if current_landmarks:
            for landmarks in current_landmarks:
                # Check for raised hands
                # In MediaPipe, indices for important landmarks:
                # 0: nose, 11: left shoulder, 12: right shoulder, 13: left elbow,
                # 14: right elbow, 15: left wrist, 16: right wrist

                # Check if wrists are above shoulders (raised arms)
                if landmarks[15][2] > 0.5 and landmarks[11][2] > 0.5:  # Left wrist and shoulder visible
                    if landmarks[15][1] < landmarks[11][1]:  # Y-coordinate comparison (higher is smaller y)
                        fight_indicators.append("Left arm raised")
                        # Mark raised left hand
                        cv2.circle(viz_frame, (landmarks[15][0], landmarks[15][1]), 10, (0, 0, 255), -1)

                if landmarks[16][2] > 0.5 and landmarks[12][2] > 0.5:  # Right wrist and shoulder visible
                    if landmarks[16][1] < landmarks[12][1]:  # Y-coordinate comparison
                        fight_indicators.append("Right arm raised")
                        # Mark raised right hand
                        cv2.circle(viz_frame, (landmarks[16][0], landmarks[16][1]), 10, (0, 0, 255), -1)

                # Analyze arm angles for punching posture
                # Calculate angle between shoulder, elbow and wrist
                if all(landmarks[i][2] > 0.5 for i in [11, 13, 15]):  # Left arm landmarks visible
                    left_angle = self._calculate_angle(
                        (landmarks[11][0], landmarks[11][1]),  # Left shoulder
                        (landmarks[13][0], landmarks[13][1]),  # Left elbow
                        (landmarks[15][0], landmarks[15][1])  # Left wrist
                    )
                    if left_angle < 90:  # Bent arm - potential punching position
                        fight_indicators.append("Left arm bent")

                if all(landmarks[i][2] > 0.5 for i in [12, 14, 16]):  # Right arm landmarks visible
                    right_angle = self._calculate_angle(
                        (landmarks[12][0], landmarks[12][1]),  # Right shoulder
                        (landmarks[14][0], landmarks[14][1]),  # Right elbow
                        (landmarks[16][0], landmarks[16][1])  # Right wrist
                    )
                    if right_angle < 90:  # Bent arm - potential punching position
                        fight_indicators.append("Right arm bent")

        # 3. Analyze motion between frames (fast movements may indicate fighting)
        if self.prev_poses and current_landmarks:
            for prev_landmarks in self.prev_poses:
                for curr_landmarks in current_landmarks:
                    # Check wrist movement speed (index 15 and 16 are wrists)
                    for wrist_idx in [15, 16]:
                        if prev_landmarks[wrist_idx][2] > 0.5 and curr_landmarks[wrist_idx][2] > 0.5:
                            # Calculate displacement normalized by frame size
                            displacement = distance.euclidean(
                                (prev_landmarks[wrist_idx][0], prev_landmarks[wrist_idx][1]),
                                (curr_landmarks[wrist_idx][0], curr_landmarks[wrist_idx][1])
                            ) / max(frame_width, frame_height)

                            if displacement > self.limb_speed_threshold:
                                fight_indicators.append(f"Fast wrist movement ({wrist_idx})")
                                # Draw motion vector
                                cv2.arrowedLine(viz_frame,
                                                (prev_landmarks[wrist_idx][0], prev_landmarks[wrist_idx][1]),
                                                (curr_landmarks[wrist_idx][0], curr_landmarks[wrist_idx][1]),
                                                (255, 0, 0), 2)

        # Determine if a fight is happening based on multiple indicators
        current_detection = len(fight_indicators) >= 2

        # Update detection memory
        self.detection_memory.append(current_detection)
        if len(self.detection_memory) > self.memory_length:
            self.detection_memory.pop(0)

        # Calculate percentage of positive detections in memory
        detection_percentage = sum(self.detection_memory) / len(self.detection_memory) if self.detection_memory else 0

        # Final detection decision based on memory
        fight_detected = detection_percentage > self.detection_threshold

        # Update previous poses for next frame
        self.prev_poses = current_landmarks

        # Add visualization and annotations
        if fight_detected:
            cv2.putText(viz_frame, "FIGHT DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Add metrics to display
        cv2.putText(viz_frame, f"Fight indicators: {len(fight_indicators)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        indicator_y = 90
        for indicator in fight_indicators[:3]:  # Show at most 3 indicators
            cv2.putText(viz_frame, f"- {indicator}", (20, indicator_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            indicator_y += 20

        cv2.putText(viz_frame, f"Detection confidence: {detection_percentage:.2f}", (10, indicator_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Return a dummy score of 0.0 since we're not using scores anymore
        return viz_frame, fight_detected, 0.0

    def _calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points
        Args:
            a: first point (x, y)
            b: middle point (x, y) - vertex of the angle
            c: last point (x, y)
        Returns:
            angle in degrees
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
                np.sqrt(ba[0] ** 2 + ba[1] ** 2) * np.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)


class FightDetector:
    """
    Main fight detector class that uses pose-based detection
    """

    def __init__(self):
        self.pose_detector = PoseBasedFightDetector()
        print("Fight detection module initialized with MediaPipe pose estimation")

    def detect(self, frame):
        """
        Perform fight detection on a frame
        Returns: annotated frame, flag indicating whether a fight was detected, and a dummy score
        """
        # Use pose-based detection
        pose_frame, fight_detected, _ = self.pose_detector.detect(frame)

        # Return the frame with detections, the detection flag, and a dummy score of 0.0
        # The dummy score is kept for API compatibility with the original code
        return pose_frame, fight_detected, 0.0


# Test function
def test_fight_detector():
    detector = FightDetector()
    cap = cv2.VideoCapture(0)  # For webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, fight_detected, _ = detector.detect(frame)

        cv2.imshow("Fight Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_fight_detector()