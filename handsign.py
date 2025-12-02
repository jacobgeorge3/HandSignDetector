import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import logging
import os
from typing import List, Tuple, Optional, Dict

import classifier
import static_handsign
from cluster import get_cluster_center
from classifier import create_descriptors
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandSignDetector:
    """
    detects and classifies hand signs in real-time using MediaPipe and a custom classifier.
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
        self.bar_chart = None
        
        self.load_class_names()
        self.load_truth_data()
        self.initialize_plot()

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
                # This might cause issues if not handled, but for now we append a placeholder or handle it in classification
                # Ideally we should probably skip or fail, but let's try to be robust.
                # For now, let's assume we need valid data.
                # If we really can't find data, we might need to handle it. 
                # Let's just append a zero vector or similar if needed, but the original code assumed valid data.
                # We will skip appending to truth_descriptors and handle index mismatch if any.
                # Actually, to keep indices aligned with class_names, we must append something.
                self.truth_descriptors.append(np.zeros(20)) 
            else:
                truth_data = truth_data.squeeze()
                # Handle case where squeeze results in 1D array if only one sample
                if truth_data.ndim == 1:
                     truth_data = np.expand_dims(truth_data, axis=0)
                     
                descriptors = create_descriptors(truth_data)
                if descriptors.size > 0:
                    cluster_center = get_cluster_center(descriptors)
                    self.truth_descriptors.append(cluster_center)
                else:
                     self.truth_descriptors.append(np.zeros(20))

    def initialize_plot(self):
        """Initialize the matplotlib bar chart."""
        plt.ion()
        self.fig = plt.figure()
        self.bar_chart = plt.bar(self.class_names, [0] * len(self.class_names))
        plt.ylim(bottom=0, top=10)
        plt.yticks(np.arange(0, 11, step=1))

    def update_plot(self):
        """Update the bar chart with current detection counts."""
        if self.bar_chart:
            for i, rect in enumerate(self.bar_chart):
                rect.set_height(self.signs_count[i])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect hands, classify gestures, and draw results.
        
        Args:
            frame: The input video frame.
            
        Returns:
            The processed frame with annotations.
        """
        x, y, c = frame.shape
        
        # Flip frame vertically (mirror view)
        frame = cv2.flip(frame, 1)
        
        imagergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imagergb)
        
        class_name_str = ""
        self.signs_count = [0] * len(self.class_names) # Reset counts for this frame? 
        # Original code accumulated counts over time? 
        # "signs = np.zeros(5).astype(int).tolist()" was inside the loop in original code, 
        # but "bar[i].set_height(signs[i])" used it.
        # Wait, in original code:
        # while True:
        #   signs = np.zeros(5).astype(int).tolist()
        #   if result.multi_hand_landmarks:
        #       for handslms in result.multi_hand_landmarks:
        #           ...
        #           signs[np.argmin(dists)] += 1
        #   ...
        #   for i in range(len(signs)): bar[i].set_height(signs[i])
        
        # So it resets every frame.
        
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
        
        # Construct display string
        display_parts = []
        for i, name in enumerate(self.class_names):
            display_parts.append(f"{name}: {self.signs_count[i]}")
        class_name_str = " ".join(display_parts)

        # Show prediction on frame
        cv2.putText(frame, class_name_str, config.TEXT_POSITION, 
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 
                    config.TEXT_COLOR, config.FONT_THICKNESS, cv2.LINE_AA)
        
        return frame

    def run(self):
        """Run the main webcam loop."""
        # Use camera index 1 for built-in camera (0 is often Continuity Camera on macOS)
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            logging.error("Could not open webcam.")
            return

        logging.info("Starting webcam feed. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read frame.")
                    break
                
                frame = self.process_frame(frame)
                cv2.imshow("Output", frame)
                
                self.update_plot()
                
                # Check for 'q' or 'Q' key, or ESC key (27)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    break
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.close('all')

if __name__ == "__main__":
    detector = HandSignDetector()
    detector.run()