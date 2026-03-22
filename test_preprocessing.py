import cv2
import numpy as np
import sys
import os

# Add modules folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from preprocessing import preprocess_image, clahe_enhancement, histogram_equalization

print("🎥 Preprocessing Test - Press 'q' to quit")
print("Keys: 'd' = darken image, 'n' = normal, 'b' = brighten\n")

cap = cv2.VideoCapture(0)
mode = "normal"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate different lighting conditions
    if mode == "dark":
        # Simulate dark room
        test_frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=-50)
    elif mode == "bright":
        # Simulate overexposed
        test_frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=80)
    else:
        test_frame = frame.copy()

    # Apply preprocessing
    processed = preprocess_image(test_frame.copy())

    # Show brightness values
    gray_before = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    brightness_before = np.mean(gray_before)
    brightness_after = np.mean(gray_after)

    # Add labels
    cv2.putText(test_frame, f"BEFORE - Brightness: {brightness_before:.0f}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(processed, f"AFTER - Brightness: {brightness_after:.0f}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(test_frame, f"Mode: {mode.upper()}",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show both side by side
    combined = np.hstack([test_frame, processed])
    cv2.imshow("Preprocessing Test (Before vs After)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        mode = "dark"
        print("🌙 Switched to DARK mode")
    elif key == ord('n'):
        mode = "normal"
        print("✅ Switched to NORMAL mode")
    elif key == ord('b'):
        mode = "bright"
        print("☀️ Switched to BRIGHT mode")

cap.release()
cv2.destroyAllWindows()
print("Preprocessing test complete!")