import cv2
import numpy as np
from collections import deque
import math
import smbus
import time

# ========== I2C Initialization ==========
bus = smbus.SMBus(1)
arduino_address = 0x08
SEND_INTERVAL = 0.05  # 50ms between sends

# ========== Coordinate Scaling ==========
PI_WIDTH = 1280  # Match camera resolution
PI_HEIGHT = 720
ARDUINO_RANGE = 230  # Scaled range for Arduino

def int16_to_bytes(val):
    """Proper 16-bit signed integer conversion"""
    val = int(val)
    return [(val >> 8) & 0xFF, val & 0xFF]

# ========== Smoothing Buffer ==========
POSITION_BUFFER_SIZE = 7
position_buffer = deque(maxlen=POSITION_BUFFER_SIZE)

def smooth_position(current_pos):
    """Apply moving average filter"""
    position_buffer.append(current_pos)
    if len(position_buffer) >= 3:
        return (
            int(np.mean([p[0] for p in position_buffer])),
            int(np.mean([p[1] for p in position_buffer]))
        )
    return current_pos

# ========== Tracking Parameters ==========
lower_red_1 = np.array([0, 100, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 50])
upper_red_2 = np.array([180, 255, 255])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# ========== Camera Pipeline ==========
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=60/1 ! "
    "jpegdec ! videoconvert ! appsink"
)
video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# ========== Tracking State ==========
tracking_roi = None
position_history = deque(maxlen=10)
visited_positions = set()
last_send_time = time.time()
MIN_DISTANCE_THRESHOLD = 60  # Pixels to consider new target

def calculate_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def detect_black_curve(roi, roi_position):
    # ... (keep your existing curve detection code) ...
    return current_intersections, mask

# ========== Improved Target Selection ==========
def select_best_target(intersections, current_pos, visited):
    """Choose target point with smart validation"""
    if not intersections or not current_pos:
        return None

    # Filter nearby and recently visited points
    valid = []
    for ip in intersections:
        dist = calculate_distance(ip, current_pos)
        if dist > MIN_DISTANCE_THRESHOLD and ip not in visited:
            valid.append(ip)

    # Select farthest valid point
    if valid:
        return max(valid, key=lambda p: calculate_distance(p, current_pos))
    return None

# ========== Main Loop ==========
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # ===== Ball Detection =====
    detected = False
    current_pos = None

    # ... (keep your existing ball detection logic) ...
    # After detecting center_x, center_y:

    if detected:
        # Apply smoothing
        smoothed_pos = smooth_position((center_x, center_y))
        position_history.append(smoothed_pos)
        current_pos = smoothed_pos

    # ===== Target Selection =====
    desired_point = None
    if current_pos and intersection_points:
        desired_point = select_best_target(intersection_points, 
                                         current_pos, 
                                         visited_positions)

    # ===== I2C Communication =====
    if current_pos and desired_point:
        now = time.time()
        if now - last_send_time >= SEND_INTERVAL:
            try:
                # Scale coordinates for Arduino
                scaled_current = (
                    int(current_pos[0] * ARDUINO_RANGE / PI_WIDTH),
                    int(current_pos[1] * ARDUINO_RANGE / PI_HEIGHT)
                )
                scaled_desired = (
                    int(desired_point[0] * ARDUINO_RANGE / PI_WIDTH),
                    int(desired_point[1] * ARDUINO_RANGE / PI_HEIGHT)
                )

                # Prepare I2C data (4 x int16)
                data = []
                for val in [scaled_current[0], scaled_current[1],
                            scaled_desired[0], scaled_desired[1]]:
                    data.extend(int16_to_bytes(val))

                bus.write_i2c_block_data(arduino_address, 0, data)
                last_send_time = now

            except Exception as e:
                print(f"I2C Error: {e}")

    # ===== Visualization =====
    if current_pos:
        cv2.circle(frame, current_pos, 8, (0,0,255), -1)
    if desired_point:
        cv2.arrowedLine(frame, current_pos, desired_point, 
                       (255,0,255), 3, tipLength=0.3)
        
    # ... (keep your existing visualization code) ...

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
