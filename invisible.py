import cv2
import numpy as np
import time

# Open webcam
cap = cv2.VideoCapture(0)
time.sleep(3)

# Capture the background (60 frames to reduce noise)
print("Capturing background... Please move out of the frame!")
for i in range(60):
    ret, background = cap.read()
    if not ret:
        continue
background = np.flip(background, axis=1)

print("Background captured! Wear your cloak and see the magic 🪄")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural view
    frame = np.flip(frame, axis=1)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 🎯 Define the cloak color range (light green here, adjust if needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    # Mask for green color
    mask1 = cv2.inRange(hsv, lower_green, upper_green)

    # Clean mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    # Segment cloak and non-cloak areas
    cloak_area = cv2.bitwise_and(background, background, mask=mask1)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask2)

    # Combine
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    # Show
    cv2.imshow("🧥 Invisible Cloak Effect", final_output)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
