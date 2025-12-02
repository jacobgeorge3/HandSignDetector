import numpy as np
import random
from typing import List, Tuple, Union
from sklearn.metrics import euclidean_distances
import config

def create_descriptor(landmarks: List[Tuple[float, float]]) -> Union[np.ndarray, List]:
  """
  Generates a descriptor from a list of landmarks.

  Args:
      landmarks: A list of 21 (x, y) tuples representing hand landmarks.
                 Values should be normalized to [0, 1].

  Returns:
      A 20-element numpy array representing normalized distances from the wrist (base),
      or an empty list if input is invalid.
  """
  if len(landmarks) < 21:
    return []
  
  # The base here will be considered as the WRIST point (index 0)
  base_x, base_y = landmarks[0] 
  lm_dists_normed = []
  
  # Compute distance between WRIST base and every other point
  for i in range(20):
    lm_x, lm_y = landmarks[i+1]
    # Euclidean distance: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    lm_dist = np.sqrt(np.square(lm_x - base_x) + np.square(lm_y - base_y))
    lm_dists_normed.append(lm_dist)

  # Normalization: divide each distance by the largest distance measured
  max_dist = max(lm_dists_normed) if lm_dists_normed else 1.0
  lm_dists_normed = np.array(lm_dists_normed)
  if max_dist > 0:
      lm_dists_normed /= max_dist
      
  return lm_dists_normed


def create_descriptors(hands: List[List[Tuple[float, float]]]) -> np.ndarray:
  """
  Generates descriptors for multiple hands.

  Args:
      hands: A list of hands, where each hand is a list of landmarks.

  Returns:
      A numpy array of descriptors.
  """
  descriptors = []
  for hand_lm in hands:
    descriptor = create_descriptor(hand_lm)
    if len(descriptor) > 0:
        descriptors.append(descriptor)
  return np.array(descriptors)


def dist_to_target(descriptor: np.ndarray, target_descriptor: np.ndarray) -> float:
  """
  Computes a pseudo-Euclidean distance between a descriptor and a target descriptor.
  Certain indices are penalized more heavily.

  Args:
      descriptor: 20-d descriptor of normalized dists to base point.
      target_descriptor: 20-d descriptor for the target class.

  Returns:
      The computed distance, or -1 if inputs are incompatible.
  """
  if len(descriptor) != len(target_descriptor):
    return -1.0  # error code

  euclidean_dist = 0.0
  for i in range(len(descriptor)):
    diff_sq = np.square(descriptor[i] - target_descriptor[i])
    if i in config.PENALTY_INDICES:
      diff_sq *= config.PENALTY_FACTOR
    euclidean_dist += diff_sq
    
  return np.sqrt(euclidean_dist)


## TESTING FUNCTIONS
def get_fake_hand_points() -> List[Tuple[float, float]]:
  """Generates a fake hand with random points for testing."""
  fake_hand_points = []
  for i in range(21):
    x = random.randint(0,100) / 100.0
    y = random.randint(0,100) / 100.0
    fake_hand_points.append((x,y))
  return fake_hand_points

def test_funcs():
  fake_non_target = get_fake_hand_points() # a new hand
  descriptor = create_descriptor(fake_non_target)

  fake_targets = []
  for i in range(5):
    fake_targets.append(create_descriptor(get_fake_hand_points()))

  for fake_targ in fake_targets:  # existing hands
    print(dist_to_target(descriptor, fake_targ))