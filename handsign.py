import cv2
import numpy as np
import mediapipe as mp
import logging
import os
import time
from typing import List, Tuple, Optional

import classifier
import static_handsign
from cluster import get_cluster_center
from classifier import create_descriptors
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandSignDetector:
    """
    Detects and classifies hand signs in real-time using MediaPipe and a custom classifier.
    """

    def __init__(self):
        """Initialize the HandSignDetector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.class_names: List[str] = []
        self.truth_descriptors: List[np.ndarray] = []
        self.signs_count: List[int] = []
        
        self.load_class_names()
        self.load_truth_data()

    def load_class_names(self):
        """Load gesture names from file."""
        try:
            with open(config.GESTURE_NAMES_FILE, 'r') as f:
                self.class_names = f.read().splitlines()
            logging.info(f"Loaded class names: {self.class_names}")
            self.signs_count = [0] * len(self.class_names)
        except FileNotFoundError:
            logging.error(f"Class names file not found: {config.GESTURE_NAMES_FILE}")
            self.class_names = []

    def load_truth_data(self):
        """Load and process truth data for each gesture."""
        self.truth_descriptors = []
        
        if not self.class_names:
            logging.warning("No class names loaded. Skipping truth data loading.")
            return

        for name in self.class_names:
            image_path = os.path.join(config.IMAGES_DIR, name.lower().replace(" ", ""))
            logging.info(f"Loading truth data for {name} from {image_path}")
            
            truth_data = static_handsign.generate_truth_data(image_path)
            
            if truth_data.size == 0:
                logging.warning(f"No truth data found for {name}. Using empty descriptor.")
                self.truth_descriptors.append(np.zeros(20)) 
            else:
                truth_data = truth_data.squeeze()
                if truth_data.ndim == 1:
                     truth_data = np.expand_dims(truth_data, axis=0)
                     
                descriptors = create_descriptors(truth_data)
                if descriptors.size > 0:
                    cluster_center = get_cluster_center(descriptors)
                    self.truth_descriptors.append(cluster_center)
                else:
                     self.truth_descriptors.append(np.zeros(20))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Process a single frame: detect hands, classify gestures, and draw results.
        
        Args:
            frame: The input video frame.
            
        Returns:
            Tuple of (processed frame with annotations, list of gesture counts)
        """
        x, y, c = frame.shape
        
        # Flip frame vertically (mirror view)
        frame = cv2.flip(frame, 1)
        
        imagergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imagergb)
        
        self.signs_count = [0] * len(self.class_names)
        
        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                landmarks = []
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(frame, handslms, self.mp_hands.HAND_CONNECTIONS)
                
                # Classify
                new_descriptor = classifier.create_descriptor(landmarks)
                
                # Calculate distances to all truth descriptors
                dists = []
                for target_desc in self.truth_descriptors:
                    dists.append(classifier.dist_to_target(new_descriptor, target_desc))
                
                # Find best match
                min_dist_idx = np.argmin(dists)
                
                # Ratio test
                sorted_dists = sorted(dists)
                if len(sorted_dists) >= 2:
                    ratio = sorted_dists[0] / sorted_dists[1] if sorted_dists[1] > 0 else 0
                    
                    if ratio < config.RATIO_TEST_THRESHOLD:
                        current_class = self.class_names[min_dist_idx]
                    else:
                        current_class = ""
                else:
                    current_class = self.class_names[min_dist_idx]

                # Increment count
                self.signs_count[min_dist_idx] += 1
        
        return frame, self.signs_count


class HandSignGUI:
    """Custom OpenCV GUI for Hand Sign Detection."""
    
    def __init__(self):
        """Initialize the GUI."""
        self.detector = HandSignDetector()
        self.cap = None
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # UI Constants
        self.SIDEBAR_WIDTH = 300
        self.BUTTON_HEIGHT = 60
        self.BUTTON_COLOR = (54, 67, 244) # Red in BGR
        self.BUTTON_HOVER_COLOR = (47, 47, 211)
        self.BG_COLOR = (43, 43, 43) # Dark gray
        self.TEXT_COLOR = (255, 255, 255)
        self.ACCENT_COLOR = (80, 175, 76) # Green
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.mouse_click = False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click = True

    def draw_dashboard(self, frame: np.ndarray, gesture_counts: List[int]) -> np.ndarray:
        """Draw the dashboard overlay on the frame."""
        h, w, c = frame.shape
        
        # Create a new image with sidebar
        new_w = w + self.SIDEBAR_WIDTH
        canvas = np.zeros((h, new_w, c), dtype=np.uint8)
        canvas[:, :w] = frame
        
        # Draw sidebar background
        canvas[:, w:] = self.BG_COLOR
        
        # Draw Title
        cv2.putText(canvas, "Hand Sign", (w + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Detector", (w + 20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.TEXT_COLOR, 2, cv2.LINE_AA)
        
        # Draw Separator
        cv2.line(canvas, (w + 20, 110), (new_w - 20, 110), (100, 100, 100), 1)
        
        # Draw Gestures
        y_offset = 150
        for i, name in enumerate(self.detector.class_names):
            count = gesture_counts[i]
            color = self.ACCENT_COLOR if count > 0 else (150, 150, 150)
            
            # Gesture Name
            cv2.putText(canvas, name.capitalize(), (w + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 1, cv2.LINE_AA)
            
            # Count
            cv2.putText(canvas, str(count), (new_w - 50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            y_offset += 40
            
        # Draw FPS
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (w + 20, h - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
            
        # Draw Quit Button
        btn_x1 = w + 20
        btn_y1 = h - 80
        btn_x2 = new_w - 20
        btn_y2 = h - 20
        
        # Check hover
        mx, my = self.mouse_pos
        is_hover = (btn_x1 <= mx <= btn_x2) and (btn_y1 <= my <= btn_y2)
        btn_color = self.BUTTON_HOVER_COLOR if is_hover else self.BUTTON_COLOR
        
        cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
        cv2.putText(canvas, "QUIT", (w + 110, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2, cv2.LINE_AA)
        
        # Handle click
        if self.mouse_click and is_hover:
            self.running = False
            
        self.mouse_click = False # Reset click
        
        return canvas

    def run(self):
        """Run the main application loop."""
        logging.info("Attempting to open camera index 1...")
        self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            logging.error("Could not open webcam.")
            return

        logging.info("Camera opened successfully.")
        self.running = True
        
        # Create window and set mouse callback
        cv2.namedWindow("Hand Sign Detector")
        cv2.setMouseCallback("Hand Sign Detector", self.mouse_callback)
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to read frame.")
                    break
                
                # Process frame
                processed_frame, gesture_counts = self.detector.process_frame(frame)
                
                # Draw dashboard
                final_frame = self.draw_dashboard(processed_frame, gesture_counts)
                
                # Show frame
                cv2.imshow("Hand Sign Detector", final_frame)
                
                # Calculate FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Check for 'q' key or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    self.running = False
                    
                # Check if window was closed
                if cv2.getWindowProperty("Hand Sign Detector", cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False

        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandSignGUI()
    app.run()