import cv2
import mediapipe as mp
import numpy as np
import os
import logging
from typing import List, Optional
import config

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def generate_truth_data(directory: str) -> np.ndarray:
  """
  Generates coordinates for training data from images in a directory.

  Args:
      directory (str): Path to the directory holding the training images.

  Returns:
      np.ndarray: nx21x2 np array, with n being number of images.
                  Returns empty array if directory doesn't exist or no hands found.
  """
  if not os.path.exists(directory):
      logging.error(f"Directory not found: {directory}")
      return np.array([])

  with mp_hands.Hands(
      static_image_mode=config.STATIC_IMAGE_MODE,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
    
    hand_data = []
    
    # Iterate over files in directory
    try:
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
                
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.imread(entry.path)
            if image is None:
                logging.warning(f"Could not read image: {entry.path}")
                continue
                
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue

            image_height, image_width, _ = image.shape
            
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_data = []
                for lm in hand_landmarks.landmark:
                    lmx = int(lm.x * image_width)
                    lmy = int(lm.y * image_height)
                    landmark_data.append([lmx, lmy])
                hand_data.append(landmark_data)
                
    except Exception as e:
        logging.error(f"Error processing directory {directory}: {e}")

  return np.array(hand_data)
