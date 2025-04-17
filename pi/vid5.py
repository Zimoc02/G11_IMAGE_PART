import cv2
import numpy as np

# Define HSV color ranges for red (baby plum tomato color)
lower_red_1 = np.array([0, 150, 100])   # Light red shades
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 150, 100])  # Deep red shades
upper_red_2 = np.array([180, 255, 255])

# Initialize the camera
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red color detection
    mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)

    # Combine both masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects/noise
            (x, y, w, h) = cv2.boundingRect(contour)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw a circle at the center of the object
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Print the position (X, Y)
            print(f"Position -> X: {center_x}, Y: {center_y}")

            # Show position on the video feed
            cv2.putText(frame, f"({center_x}, {center_y})", (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frames
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Tracking Red Object", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
