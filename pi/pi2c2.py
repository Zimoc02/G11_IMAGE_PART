import cv2
import numpy as np
from collections import deque
import math
import smbus
import pandas as pd

# I2C Setup
bus = smbus.SMBus(1)
arduino_address = 0x08
SCALING_FACTOR = 0.5  # Start with 0.5 if original path was for 1280x720 and camera is 640x480
OFFSET_X = 30         # Adjust horizontal positioning
OFFSET_Y = 20 

def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val
    return [(val >> 8) & 0xFF, val & 0xFF]

def send_two_points_16bit(x1, y1, x2, y2):
    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}), ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")

def load_path_coordinates(csv_file):
    df = pd.read_csv(csv_file, header=None)
    raw_coords = df.values.astype(float)
    
    # Apply scaling and offset
    scaled_coords = raw_coords * SCALING_FACTOR
    scaled_coords[:, 0] += OFFSET_X  # X offset
    scaled_coords[:, 1] += OFFSET_Y  # Y offset
    
    return scaled_coords.astype(int)

# Color ranges for ball detection
lower_red_1 = np.array([0, 150, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 150, 100])
upper_red_2 = np.array([180, 255, 255])

# Initialize camera
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 60)

# Tracking parameters
TRACKING_SIZE = 70
MIN_CONTOUR_AREA = 200
LOOKAHEAD_POINTS = 15  # How many path points ahead to target

# Tracking variables
tracking_roi = None
position_history = deque(maxlen=10)
last_ball_pos = None

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

def overlay_path(frame, path):
    for i in range(len(path)-1):
        start = tuple(path[i].astype(int))
        end = tuple(path[i+1].astype(int))
        cv2.line(frame, start, end, (0, 255, 0), 2)
    return frame

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = overlay_path(frame, path_coordinates)
    detected = False

    if tracking_roi is not None:
        x, y, w, h = tracking_roi
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w > 0 and h > 0:
            roi = frame[y:y+h, x:x+w]
            
            # Detect red ball
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv_roi, lower_red_1, upper_red_1),
                cv2.inRange(hsv_roi, lower_red_2, upper_red_2)
            )
            
            kernel = np.ones((5,5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                    (x_local, y_local, w_local, h_local) = cv2.boundingRect(contour)
                    center_x = x + x_local + w_local//2
                    center_y = y + y_local + h_local//2
                    
                    new_x = center_x - TRACKING_SIZE//2
                    new_y = center_y - TRACKING_SIZE//2
                    new_x = max(0, min(new_x, frame.shape[1] - TRACKING_SIZE))
                    new_y = max(0, min(new_y, frame.shape[0] - TRACKING_SIZE))
                    tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                    
                    position_history.append((center_x, center_y))
                    last_ball_pos = (center_x, center_y)
                    detected = True
                    break

    if not detected:
        tracking_roi = None
        # Full frame search if tracking lost
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_frame, lower_red_1, upper_red_1),
            cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        )
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                (x, y, w, h) = cv2.boundingRect(contour)
                center_x = x + w//2
                center_y = y + h//2
                
                new_x = center_x - TRACKING_SIZE//2
                new_y = center_y - TRACKING_SIZE//2
                new_x = max(0, min(new_x, frame.shape[1] - TRACKING_SIZE))
                new_y = max(0, min(new_y, frame.shape[0] - TRACKING_SIZE))
                tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                break

    # Path following logic
    # Path following logic
    if last_ball_pos is not None:
        current_x, current_y = last_ball_pos
    
        # Find nearest path point
        distances = np.linalg.norm(path_coordinates - (current_x, current_y), axis=1)
        nearest_idx = np.argmin(distances)
    
        # Look ahead on the path
        target_idx = min(nearest_idx + LOOKAHEAD_POINTS, len(path_coordinates)-1)
        target_point = path_coordinates[target_idx].astype(int)  # Convert to integers
    
        try:
            # Send coordinates to Arduino
            send_two_points_16bit(current_x, current_y, target_point[0], target_point[1])
        
            # Draw navigation elements
            cv2.circle(frame, (current_x, current_y), 7, (255, 0, 0), -1)
            cv2.circle(frame, tuple(target_point), 7, (0, 0, 255), -1)  # Already converted to int
            cv2.arrowedLine(frame, 
                        (int(current_x), int(current_y)),  # Explicit integer conversion
                        tuple(target_point), 
                        (255, 0, 255), 2)
        except Exception as e:
            print(f"Drawing error: {e}")
            # Reset tracking if invalid coordinates
            tracking_roi = None
            last_ball_pos = None
            continue

    # Draw ROI box if available
    if tracking_roi is not None:
        x_roi, y_roi, w_roi, h_roi = tracking_roi
        cv2.rectangle(frame, (x_roi, y_roi), 
                     (x_roi + w_roi, y_roi + h_roi), (0, 255, 255), 2)

    cv2.imshow("Path Following", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
