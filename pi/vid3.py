import cv2
import numpy as np
import time

# Define HSV color range for the object (Adjust these values)
# Use an HSV color picker to find the correct values for your AirPod
lower_hsv = np.array([0, 0, 200])  # Example for white-ish objects
upper_hsv = np.array([180, 50, 255])

# Initialize video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Initial variables for tracking
prev_position = None
prev_time = time.time()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask based on the defined HSV range
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    # Find contours of the masked object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Filter small noise
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate center of the object
            current_position = np.array([x + w / 2, y + h / 2])

            # Calculate motion vectors
            if prev_position is not None:
                current_time = time.time()
                dt = current_time - prev_time

                # Velocity and acceleration
                velocity = (current_position - prev_position) / dt
                acceleration = velocity / dt

                print(f"Velocity: {velocity}, Acceleration: {acceleration}")

                prev_time = current_time
                prev_position = current_position
            else:
                prev_position = current_position

    # Display the mask and the detection frame
    cv2.imshow("Mask", mask)
    cv2.imshow("Object Tracking", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
