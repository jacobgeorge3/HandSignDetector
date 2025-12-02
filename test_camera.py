import cv2
import sys

print("Testing camera access...")

# Try different camera indices
for i in range(3):
    print(f"\nTrying camera index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        print(f"  ✓ Camera {i} opened successfully")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Successfully read frame from camera {i}")
            print(f"  Frame shape: {frame.shape}")
            cap.release()
            print(f"\n✅ Camera {i} is working! Use this index.")
            sys.exit(0)
        else:
            print(f"  ✗ Could not read frame from camera {i}")
            cap.release()
    else:
        print(f"  ✗ Could not open camera {i}")

print("\n❌ No working cameras found")
