import unittest
import numpy as np
import cv2
from handsign import HandSignDetector
import config

class TestHandSignDetector(unittest.TestCase):
    def setUp(self):
        # Initialize detector
        # Note: This might fail if it tries to open a window or plot, 
        # but we refactored to separate initialization.
        # However, __init__ calls initialize_plot which calls plt.figure()
        # This is fine for local execution usually.
        self.detector = HandSignDetector()

    def test_initialization(self):
        self.assertIsNotNone(self.detector.hands)
        self.assertIsNotNone(self.detector.class_names)
        # Check if class names were loaded (assuming gesture.names exists)
        if self.detector.class_names:
            self.assertEqual(len(self.detector.class_names), len(self.detector.truth_descriptors))

    def test_process_frame_no_hands(self):
        # Create a blank black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_frame = self.detector.process_frame(frame)
        
        # Check if frame is returned and has same shape
        self.assertEqual(processed_frame.shape, (480, 640, 3))
        
        # Check if signs_count is all zeros (no hands detected)
        self.assertEqual(sum(self.detector.signs_count), 0)

    def test_process_frame_with_mock_hand(self):
        # It's hard to mock a hand that MediaPipe detects without a real image.
        # But we can at least run the pipeline and ensure no crash.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white rectangle to simulate *something*, though MP won't detect it as a hand
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
        
        try:
            processed_frame = self.detector.process_frame(frame)
            self.assertIsNotNone(processed_frame)
        except Exception as e:
            self.fail(f"process_frame raised exception: {e}")

if __name__ == '__main__':
    unittest.main()
