import cv2
import numpy as np

# Load path coordinates from maze solver
path_coordinates = np.column_stack((path_x, path_y))  # From your first code

# Your existing ball detection code
lower_red_1 = np.array([0, 150, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 150, 100])
upper_red_2 = np.array([180, 255, 255])

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = video_capture.read()
    if not ret: break

    # Your existing ball detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            center_x = int(x + w/2)
            center_y = int(y + h/2)

            # ---------------------------------------------------------
            # NEW: Path comparison logic
            # 1. Find nearest point on path
            ball_pos = np.array([center_x, center_y])
            distances = np.linalg.norm(path_coordinates - ball_pos, axis=1)
            nearest_idx = np.argmin(distances)
            
            # 2. Get next target point (1 step ahead in path array)
            target_idx = min(nearest_idx + 1, len(path_coordinates)-1)
            target_point = path_coordinates[target_idx]
            
            # 3. Calculate position difference
            error_x = target_point[0] - center_x
            error_y = target_point[1] - center_y
            # ---------------------------------------------------------

            # Draw visualization
            cv2.circle(frame, tuple(target_point.astype(int)), 7, (0,0,255), -1)  # Target
            cv2.arrowedLine(frame, (center_x, center_y), 
                          (center_x + error_x, center_y + error_y), 
                          (255,0,0), 2)  # Error vector

            # Your existing drawing
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255,0,0), -1)
            cv2.putText(frame, f"Error: ({error_x}, {error_y})", (center_x+10, center_y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Print raw errors for motor control
            print(f"X_diff: {error_x}, Y_diff: {error_y}")

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
