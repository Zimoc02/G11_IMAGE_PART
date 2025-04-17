import cv2
import numpy as np
import time

# Initialize the camera
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Read the first frame
ret, prev_frame = video_capture.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Time tracking for acceleration
prev_time = time.time()
prev_motion_vector = np.array([0, 0])

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the difference between frames
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Find contours of the moving object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        # Calculate the bounding box
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute motion vector (displacement)
        motion_vector = np.array([x + w / 2, y + h / 2])

        # Calculate time difference
        current_time = time.time()
        dt = current_time - prev_time

        # Calculate velocity vector
        velocity = (motion_vector - prev_motion_vector) / dt

        # Calculate acceleration vector
        acceleration = (velocity - prev_motion_vector) / dt

        print(f"Velocity: {velocity}, Acceleration: {acceleration}")

        # Update for the next frame
        prev_motion_vector = velocity
        prev_time = current_time

    # Show the frame with motion detection
    cv2.imshow("Motion Detection", frame)

    # Update previous frame
    prev_gray = gray.copy()

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
