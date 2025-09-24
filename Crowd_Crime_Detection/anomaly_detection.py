import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import math
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetector:
    def __init__(self, confidence_threshold=0.25, iou_threshold=0.45):
        # Initialize detection model
        try:
            self.model = YOLO("yolov8m.pt")
            print("Using YOLOv8m model for better detection")
        except:
            self.model = YOLO("yolov8n.pt")
            print("Using YOLOv8n model")

        # Initialize tracker
        self.tracker = sv.ByteTrack()

        # Initialize box annotator
        try:
            self.box_annotator = sv.BoxAnnotator(thickness=2, text_scale=0.5)
        except TypeError:
            self.box_annotator = sv.BoxAnnotator(thickness=2)

        # Detection parameters
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Tracking history
        self.frame_count = 0
        self.trajectories = defaultdict(lambda: deque(maxlen=90))  # 3 seconds at 30fps
        self.hand_positions = defaultdict(lambda: deque(maxlen=60))  # Hand tracking
        self.item_positions = defaultdict(lambda: deque(maxlen=60))  # Track item movement
        self.proximity_history = defaultdict(lambda: deque(maxlen=90))  # Proximity history
        self.ownership_map = {}  # Track who owns which items

        # Theft detection parameters - balanced approach
        self.proximity_threshold = 80  # Distance to consider "close" interaction
        self.pocket_region_ratio = 0.4  # Relative position for pocket height
        self.looking_around_threshold = 5  # Frames of looking around to be suspicious
        self.min_detection_confidence = 3  # Consecutive frames needed for detection

        # Detection state
        self.consecutive_detections = 0
        self.detection_cooldown = 0
        self.active_interactions = {}
        self.last_detection = None

        # Debug mode
        self.debug = True

        # Suspicious behavior tracking
        self.suspicious_behaviors = defaultdict(dict)  # Track suspicious actions by person

        # Target classes for detection
        self.target_classes = [0, 67, 26, 28, 73, 39, 41, 64, 65, 66, 77]
        self.class_names = {
            0: "person",
            26: "handbag",
            28: "backpack",
            39: "bottle",
            41: "cup",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            73: "laptop",
            77: "cell phone"
        }

        # Initialize color ranges for red detection
        self.target_colors = {
            'red': ([0, 100, 100], [10, 255, 255]),  # First range of red in HSV
            'red2': ([160, 100, 100], [180, 255, 255])  # Second range of red in HSV
        }

        print("Balanced anomaly detector initialized for retail theft detection")

    def detect(self, frame):
        """
        Detect potential theft in the given frame
        """
        # Store original frame
        original_frame = frame.copy()
        height, width = frame.shape[:2]
        self.frame_count += 1

        # Handle detection cooldown to prevent alert flooding
        if self.detection_cooldown > 0:
            self.detection_cooldown -= 1

        # Run detection
        results = self.model(frame, classes=self.target_classes, conf=self.confidence_threshold)[0]

        # Process detection results
        if len(results.boxes) == 0:
            # No detections at all
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            return frame, False, []

        # Convert to supervision Detections format
        try:
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)
        except Exception as e:
            print(f"Detection error: {e}")
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            return frame, False, []

        # Skip if no class IDs available
        if detections.class_id is None or len(detections.class_id) == 0:
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            return frame, False, []

        # Extract people and items
        people_detections = []
        item_detections = []

        for i, class_id in enumerate(detections.class_id):
            # Get tracking ID
            tracking_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            if tracking_id is None:
                continue

            # Extract box info
            box = detections.xyxy[i]
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if class_id == 0:  # Person
                # Extract person info
                person = {
                    'id': tracking_id,
                    'class_id': class_id,
                    'box': box,
                    'center': center,
                    'confidence': detections.confidence[i]
                }

                # Check for red clothing
                person['red_score'] = self._detect_color(frame, box, 'red')
                person['is_red_hoodie'] = (person['red_score'] > 0.15)

                # Add to people list
                people_detections.append(person)

                # Track person's trajectory
                self.trajectories[tracking_id].append(center)

            elif class_id in [26, 28, 39, 41, 64, 65, 66, 67, 73, 77]:  # Target items
                # Extract item info
                item = {
                    'id': tracking_id,
                    'class_id': class_id,
                    'box': box,
                    'center': center,
                    'confidence': detections.confidence[i]
                }

                # Add to items list
                item_detections.append(item)

                # Track item positions
                if tracking_id in self.item_positions:
                    # Check if item suddenly disappeared or moved
                    last_pos = self.item_positions[tracking_id][-1]
                    movement = math.sqrt((center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2)
                    item['movement'] = movement
                else:
                    item['movement'] = 0

                # Update position history
                self.item_positions[tracking_id].append(center)

        # Track item ownership
        self._update_item_possession(people_detections, item_detections)

        # Initialize detection variables
        theft_detected = False
        suspicious_regions = []
        most_suspicious_person = None
        highest_suspicion = 0

        # Check each person for suspicious behaviors
        for person in people_detections:
            person_id = person['id']
            suspicion_score = 0  # Start with zero suspicion

            # Add points for red hoodie
            if person['is_red_hoodie']:
                suspicion_score += 30

            # Check for suspicious movements
            if self._check_suspicious_movement(person_id):
                suspicion_score += 25
                # Add to tracked behaviors
                self.suspicious_behaviors[person_id]['suspicious_movement'] = True

            # Check for looking around behavior
            if self._check_looking_around(person_id):
                suspicion_score += 20
                self.suspicious_behaviors[person_id]['looking_around'] = True

            # Check for pocket interaction after having an item
            has_item = any(self.ownership_map.get(item['id']) == person_id for item in item_detections)
            pockets_checked = self._check_pocket_interaction(person, has_item)
            if pockets_checked:
                suspicion_score += 40
                self.suspicious_behaviors[person_id]['pocket_interaction'] = True

            # Check if items suddenly disappeared near this person
            for item in item_detections:
                if item['id'] in self.item_positions and len(self.item_positions[item['id']]) > 5:
                    if item['movement'] > 30 and self._is_near(person['center'], item['center'], 60):
                        suspicion_score += 25
                        self.suspicious_behaviors[person_id]['item_movement'] = True
                        break

            # Track if this is the most suspicious person
            if suspicion_score > highest_suspicion:
                highest_suspicion = suspicion_score
                most_suspicious_person = person

        # Check for item transfers between people (potential theft)
        for i, person1 in enumerate(people_detections):
            for j, person2 in enumerate(people_detections[i + 1:], i + 1):
                interaction_key = tuple(sorted([person1['id'], person2['id']]))
                distance = math.dist(person1['center'], person2['center'])

                # If people are close, track this interaction
                if distance < self.proximity_threshold:
                    # Add to proximity history
                    self.proximity_history[interaction_key].append(1)

                    # Start tracking interaction if new
                    if interaction_key not in self.active_interactions:
                        self.active_interactions[interaction_key] = {
                            'start_frame': self.frame_count,
                            'persons': [person1['id'], person2['id']],
                            'items_before': self._get_possessed_items(person1['id'], person2['id'])
                        }

                    # Check for item transfers during interaction
                    if sum(self.proximity_history[interaction_key]) > 5:
                        items_after = self._get_possessed_items(person1['id'], person2['id'])
                        items_before = self.active_interactions[interaction_key].get('items_before', {})

                        transfers = self._get_item_transfers(items_before, items_after)
                        if transfers:
                            # An item was transferred - identify recipient
                            for item_id, from_id, to_id in transfers:
                                # Find the receiver
                                receiver = None
                                for p in people_detections:
                                    if p['id'] == to_id:
                                        receiver = p
                                        break

                                if receiver:

                                    if receiver['is_red_hoodie'] or self.suspicious_behaviors[to_id]:
                                        # Add to recipient's suspicion
                                        if receiver['id'] == most_suspicious_person['id']:
                                            highest_suspicion += 30
                else:
                    # People not close - add 0 to history
                    if interaction_key in self.proximity_history:
                        self.proximity_history[interaction_key].append(0)

                    # Check if they were previously interacting but now separated
                    if interaction_key in self.active_interactions:
                        # Check if items changed hands after interaction
                        items_after = self._get_possessed_items(person1['id'], person2['id'])
                        items_before = self.active_interactions[interaction_key]['items_before']

                        transfers = self._get_item_transfers(items_before, items_after)
                        if transfers:
                            # An item was transferred - identify recipient
                            for item_id, from_id, to_id in transfers:
                                # Find the receiver
                                receiver = None
                                for p in people_detections:
                                    if p['id'] == to_id:
                                        receiver = p
                                        break

                                if receiver:

                                    if receiver['is_red_hoodie'] or self.suspicious_behaviors[to_id]:
                                        # Add to recipient's suspicion
                                        if receiver['id'] == most_suspicious_person['id']:
                                            highest_suspicion += 40

                        # Remove from active interactions
                        del self.active_interactions[interaction_key]

        # Make final theft determination based on suspicion score
        # Balanced threshold that's not too high or low
        if highest_suspicion > 60:
            # Increment consecutive detection counter
            self.consecutive_detections += 1

            # Only trigger alert after multiple consecutive detections
            if self.consecutive_detections >= self.min_detection_confidence:
                theft_detected = True

                # Create region around most suspicious person
                if most_suspicious_person:
                    x1, y1, x2, y2 = most_suspicious_person['box']
                    # Add margin around the person
                    region_x1 = max(0, int(x1) - 20)
                    region_y1 = max(0, int(y1) - 20)
                    region_width = min(width - region_x1, int(x2 - x1) + 40)
                    region_height = min(height - region_y1, int(y2 - y1) + 40)

                    suspicious_regions.append([
                        region_x1, region_y1, region_width, region_height
                    ])

                    # Set detection cooldown
                    self.detection_cooldown = 10

                    if self.debug:
                        behaviors = []
                        person_id = most_suspicious_person['id']
                        if self.suspicious_behaviors[person_id].get('suspicious_movement'):
                            behaviors.append("suspicious movement")
                        if self.suspicious_behaviors[person_id].get('looking_around'):
                            behaviors.append("looking around")
                        if self.suspicious_behaviors[person_id].get('pocket_interaction'):
                            behaviors.append("pocket interaction")
                        if self.suspicious_behaviors[person_id].get('item_movement'):
                            behaviors.append("item manipulation")
                        if most_suspicious_person['is_red_hoodie']:
                            behaviors.append("red clothing")

                        print(f"Theft detected: Person {person_id} with behaviors: {', '.join(behaviors)}")
        else:
            # Decrement consecutive detections if no suspicion
            self.consecutive_detections = max(0, self.consecutive_detections - 1)

        # Prepare visualization
        annotated_frame = self._prepare_visualization(frame, people_detections, item_detections,
                                                      suspicious_regions, theft_detected)

        return annotated_frame, theft_detected, suspicious_regions

    def _detect_color(self, frame, box, target_color):
        """Detect percentage of target color in region"""
        x1, y1, x2, y2 = [int(c) for c in box]

        # Ensure coordinates are valid
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Skip if invalid
        if x2 <= x1 or y2 <= y1:
            return 0

        # Extract region of interest
        roi = frame[y1:y2, x1:x2]

        # Convert to HSV for better color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create mask for target color
        if target_color == 'red':
            # Red in HSV wraps around 0, so need two ranges
            lower1, upper1 = self.target_colors['red']
            lower2, upper2 = self.target_colors['red2']

            mask1 = cv2.inRange(hsv_roi, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_roi, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = self.target_colors[target_color]
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))

        # Calculate percentage
        color_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1]) if roi.size > 0 else 0
        return color_ratio

    def _update_item_possession(self, people, items):
        """Update which person possesses which items based on proximity"""
        for item in items:
            if item['id'] is None:
                continue

            closest_person = None
            min_distance = float('inf')

            for person in people:
                if person['id'] is None:
                    continue

                # Calculate distance between person and item
                distance = math.dist(person['center'], item['center'])

                # Check if item is close to person
                if distance < min_distance and distance < self.proximity_threshold * 1.2:
                    min_distance = distance
                    closest_person = person['id']

            # Update ownership map
            if closest_person is not None:
                # Check if ownership changed
                if item['id'] in self.ownership_map and self.ownership_map[item['id']] != closest_person:
                    if self.debug:
                        print(
                            f"Item {item['id']} changed hands from {self.ownership_map[item['id']]} to {closest_person}")

                self.ownership_map[item['id']] = closest_person
            # Remove ownership if far from any person
            elif item['id'] in self.ownership_map and min_distance > 100:
                del self.ownership_map[item['id']]

    def _get_possessed_items(self, person1_id, person2_id):
        """Get items possessed by the specified persons"""
        possessed_items = {}
        for item_id, owner_id in self.ownership_map.items():
            if owner_id == person1_id:
                possessed_items[item_id] = person1_id
            elif owner_id == person2_id:
                possessed_items[item_id] = person2_id
        return possessed_items

    def _get_item_transfers(self, items_before, items_after):
        """Get list of transferred items with from/to information"""
        transfers = []

        # Check each item that was possessed before
        for item_id, owner_before in items_before.items():
            # If item is still tracked and has a different owner now
            if item_id in items_after and items_after[item_id] != owner_before:
                transfers.append((item_id, owner_before, items_after[item_id]))

        return transfers

    def _check_suspicious_movement(self, person_id):
        """Check for suspicious movement patterns"""
        if person_id in self.trajectories and len(self.trajectories[person_id]) >= 15:
            traj = list(self.trajectories[person_id])

            # Check for sudden direction change
            if len(traj) >= 15:
                before = traj[-15:-5]
                after = traj[-5:]

                # Calculate movement vectors
                if len(before) >= 2 and len(after) >= 2:
                    # Movement magnitude
                    before_vector = (before[-1][0] - before[0][0], before[-1][1] - before[0][1])
                    after_vector = (after[-1][0] - after[0][0], after[-1][1] - after[0][1])

                    before_mag = math.sqrt(before_vector[0] ** 2 + before_vector[1] ** 2)
                    after_mag = math.sqrt(after_vector[0] ** 2 + after_vector[1] ** 2)

                    # Check for sudden acceleration
                    if after_mag > before_mag * 1.5 and after_mag > 20:
                        return True

                    # Check for direction change
                    if before_mag > 5 and after_mag > 5:
                        # Normalize vectors
                        before_norm = (
                        before_vector[0] / before_mag, before_vector[1] / before_mag) if before_mag > 0 else (0, 0)
                        after_norm = (after_vector[0] / after_mag, after_vector[1] / after_mag) if after_mag > 0 else (
                        0, 0)

                        # Calculate dot product
                        dot_product = before_norm[0] * after_norm[0] + before_norm[1] * after_norm[1]

                        # If significant direction change
                        if dot_product < 0.5:
                            return True

        return False

    def _check_looking_around(self, person_id):
        """Check if person is looking around suspiciously"""
        # Use trajectory to detect head movement patterns
        if person_id in self.trajectories and len(self.trajectories[person_id]) >= 20:
            traj = list(self.trajectories[person_id])

            # Calculate horizontal movement variance (looking side to side)
            x_positions = [pos[0] for pos in traj[-20:]]
            x_variance = np.var(x_positions)

            # Calculate vertical movement variance (looking up/down)
            y_positions = [pos[1] for pos in traj[-20:]]
            y_variance = np.var(y_positions)

            # If more horizontal movement than vertical, might be looking around
            if x_variance > y_variance * 2 and x_variance > 40:
                return True

        return False

    def _check_pocket_interaction(self, person, has_item):
        """Check if person is interacting with pocket area (possible concealment)"""
        person_id = person['id']
        x1, y1, x2, y2 = person['box']

        # Define pocket height as percentage from top of bounding box
        pocket_y = y1 + (y2 - y1) * self.pocket_region_ratio

        # Check recent trajectory for movement toward pocket area
        if person_id in self.trajectories and len(self.trajectories[person_id]) >= 10:
            traj = list(self.trajectories[person_id])

            # Look at last few positions
            recent = traj[-5:]

            # Check if hand moved to pocket region
            for pos in recent:
                if pos[1] > pocket_y and has_item:
                    return True

        return False

    def _is_near(self, point1, point2, threshold):
        """Check if two points are within threshold distance"""
        return math.dist(point1, point2) < threshold

    def _prepare_visualization(self, frame, people, items, suspicious_regions, theft_detected):
        """Create visualization with annotations"""
        vis_frame = frame.copy()

        # Draw people bounding boxes
        for person in people:
            if person['id'] is not None:
                x1, y1, x2, y2 = [int(coord) for coord in person['box']]

                # Choose color based on red hoodie detection
                if person.get('is_red_hoodie', False):
                    color = (0, 0, 255)  # Red for red hoodie
                else:
                    color = (255, 0, 0)  # Blue for others

                # Draw person bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                # Add label with ID and suspicion indicators
                label = f"P-{person['id']}"
                if person.get('is_red_hoodie', False):
                    label += " [RED]"

                behaviors = []
                if self.suspicious_behaviors[person['id']].get('suspicious_movement'):
                    behaviors.append("M")
                if self.suspicious_behaviors[person['id']].get('looking_around'):
                    behaviors.append("L")
                if self.suspicious_behaviors[person['id']].get('pocket_interaction'):
                    behaviors.append("P")

                if behaviors:
                    label += f" [{'+'.join(behaviors)}]"

                cv2.putText(vis_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw trajectory
                if person['id'] in self.trajectories:
                    traj = list(self.trajectories[person['id']])
                    if len(traj) > 1:
                        points = np.array([(int(x), int(y)) for x, y in traj])
                        for i in range(1, len(points)):
                            cv2.line(vis_frame, tuple(points[i - 1]), tuple(points[i]), (0, 255, 0), 1)

        # Draw item bounding boxes
        for item in items:
            if item['id'] is not None:
                x1, y1, x2, y2 = [int(coord) for coord in item['box']]

                # Yellow for items
                color = (0, 255, 255)

                # Add ownership information
                owner_text = ""
                if item['id'] in self.ownership_map:
                    owner_id = self.ownership_map[item['id']]
                    owner_text = f" (P-{owner_id})"

                # Get item class name
                item_name = self.class_names.get(item['class_id'], "item")

                # Draw item box and label
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"{item_name}{owner_text}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw suspicious regions
        for x, y, w, h in suspicious_regions:
            # Red for suspicious regions
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Add theft alert if detected
        if theft_detected:
            cv2.putText(vis_frame, "POTENTIAL THEFT DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Add detection info
        cv2.putText(vis_frame, f"Detections: {self.consecutive_detections}/{self.min_detection_confidence}",
                    (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_frame

    # Set frame count from main file
    def set_frame_count(self, count):
        self.frame_count = count