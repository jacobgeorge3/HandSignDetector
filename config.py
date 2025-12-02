"""
Configuration constants for the Hand Sign Detector project.
"""

# MediaPipe Hands configuration
MAX_NUM_HANDS = 6
MIN_DETECTION_CONFIDENCE = 0.7
STATIC_IMAGE_MODE = True

# Classifier configuration
PENALTY_FACTOR = 5
PENALTY_INDICES = [3, 7, 11, 19]
RATIO_TEST_THRESHOLD = 0.4

# File paths
GESTURE_NAMES_FILE = 'gesture.names'
IMAGES_DIR = './images'

# Visualization
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 0)
TEXT_POSITION = (10, 50)
