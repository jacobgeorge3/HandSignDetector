# Hand Sign Detector

A computer vision project that detects and classifies hand signs in real-time using Google's MediaPipe and OpenCV.

## Features
- Real-time hand landmark detection
- Classification of 5 distinct hand signs:
  - Claws
  - Frogs
  - Gigem
  - Guns Up
  - Horns
- Custom descriptor based on normalized landmark distances
- "Truth data" clustering for robust classification
- Clean, modular class-based architecture

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jacobgeorge3/HandSignDetector.git
   cd HandSignDetector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the webcam feed and detection:

```bash
python handsign.py
```

Press `q` to quit the application.

## Project Structure

- `handsign.py` - Main application with `HandSignDetector` class
- `classifier.py` - Hand landmark descriptor creation and distance metrics
- `static_handsign.py` - Truth data generation from static images
- `cluster.py` - K-means clustering for gesture centroids
- `config.py` - Configuration constants
- `gesture.names` - List of gesture class names
- `images/` - Training images for each gesture
- `test_refactor.py` - Unit tests

## How it Works

The system uses MediaPipe to extract 21 hand landmarks. These landmarks are normalized and converted into a custom descriptor vector. This vector is compared against pre-computed cluster centers of "truth data" using a pseudo-Euclidean distance metric. A ratio test is applied to ensure high-confidence classifications.

## Testing

Run the test suite:

```bash
python test_refactor.py
```

## Original Project

This is a refactored version of the [CVFinalProj](https://github.com/jacobgeorge3/CVFinalProj) repository, with improved code organization, type hints, and error handling.
