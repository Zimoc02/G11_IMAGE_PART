import cv2
import numpy as np
from collections import deque
import math
import smbus
import time

# ========== I2C Configuration ==========
bus = smbus.SMBus(1)
arduino_address = 0x08
SEND_INTERVAL = 0.05  # 50ms between sends
last_send_time = time.time()

# ========== Camera Configuration ==========
PI_WIDTH = 1280
PI_HEIGHT = 720
ARDUINO_RANGE = 230  # Scaled output range for Arduino

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=60/1 ! "
    "jpegdec ! videoconvert ! appsink"
)
video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# ========== Color Ranges ==========
lower_red_1 = np.array([0, 100, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 50])
upper_red_2 = np.array([180, 255, 255])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# ========== Tracking Parameters ==========
TRACKING_SIZE = 80
CURVE_PARAMS = {'min_contour_area': 50, 'approx_epsilon': 0.02}
POSITION_BUFFER_SIZE = 5
MIN_DISTANCE_THRESHOLD = 60  # Pixels for new targets

# ========== Tracking State ==========
tracking_roi = None
position_history = deque(maxlen=10)
position_buffer = deque(maxlen=POSITION_BUFFER_SIZE)
visited_positions = set()
show_path = True

def int16_to_bytes(val):
    """Convert to 16-bit signed bytes (two's complement)"""
    val = int(val)
    return [(val >> 8) & 0xFF, val & 0xFF]

def smooth_position(current_pos):
    """Apply moving average filter with deque compatibility"""
    position_buffer.append(current_pos)
    
    # Convert deque to list for safe slicing
    buffer_list = list(position_buffer)
    if len(buffer_list) >= 3:
        recent_positions = buffer_list[-3:]
        return (
            int(np.mean([p[0] for p in recent_positions])),
            int(np.mean([p[1] for p in recent_positions]))
        )
    return current_pos

def calculate_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def detect_black_curve(roi, roi_position):
    """Detect black curve intersections in ROI"""
    x, y, w, h = roi_position
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    current_intersections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > CURVE_PARAMS['min_contour_area']:
            perimeter = cv2.arcLength(contour, True)
            epsilon = CURVE_PARAMS['approx_epsilon'] * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            for point in approx:
                px, py = point[0]
                if px == 0 or px == w-1 or py == 0 or py == h-1:
                    current_intersections.append((x + px, y + py))
    
    return current_intersections, mask

def select_best_target(intersections, current_pos):
    """Choose optimal target point with validation"""
    if not intersections or not current_pos:
        return None
    
    valid_targets = []
    for ip in intersections:
        dist = calculate_distance(ip, current_pos)
        if dist > MIN_DISTANCE_THRESHOLD and ip not in visited_positions:
            valid_targets.append(ip)
    
    if valid_targets:
        return max(valid_targets, key=lambda p: calculate_distance(p, current_pos))
    return None

# ========== Main Loop ==========
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_height, frame_width = frame.shape[:2]
    detected = False
    current_pos = None
    intersection_points = []
    desired_point = None

    # ===== Ball Detection =====
    if tracking_roi is not None:
        x, y, w, h = tracking_roi
        x, y = max(0, x), max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)
        
        if w > 0 and h > 0:
            roi = frame[y:y+h, x:x+w]
            
            # Detect red ball
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv_roi, lower_red_1, upper_red_1),
                cv2.inRange(hsv_roi, lower_red_2, upper_red_2)
            )
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    x_local, y_local, w_local, h_local = cv2.boundingRect(contour)
                    center_x = x + x_local + w_local//2
                    center_y = y + y_local + h_local//2
                    
                    # Update tracking ROI
                    new_x = max(0, center_x - TRACKING_SIZE//2)
                    new_y = max(0, center_y - TRACKING_SIZE//2)
                    tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                    
                    # Smooth position
                    smoothed_pos = smooth_position((center_x, center_y))
                    position_history.append(smoothed_pos)
                    current_pos = smoothed_pos
                    detected = True
                    
                    # Update visited positions
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            visited_positions.add((smoothed_pos[0]+dx, smoothed_pos[1]+dy))
                    
                    # Detect black curves
                    intersection_points, black_mask = detect_black_curve(roi, (x, y, w, h))
                    break

    # Fallback detection if ball lost
    if not detected:
        tracking_roi = None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_frame, lower_red_1, upper_red_1),
            cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        )
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w//2
                center_y = y + h//2
                tracking_roi = (
                    max(0, center_x - TRACKING_SIZE//2),
                    max(0, center_y - TRACKING_SIZE//2),
                    TRACKING_SIZE, TRACKING_SIZE
                )
                break

    # ===== Target Selection =====
    if current_pos and intersection_points:
        desired_point = select_best_target(intersection_points, current_pos)

    # ===== I2C Communication =====
    if current_pos and desired_point:
        now = time.time()
        if now - last_send_time >= SEND_INTERVAL:
            try:
                # Scale coordinates to Arduino range
                scaled_current = (
                    int(current_pos[0] * ARDUINO_RANGE / PI_WIDTH),
                    int(current_pos[1] * ARDUINO_RANGE / PI_HEIGHT)
                )
                scaled_desired = (
                    int(desired_point[0] * ARDUINO_RANGE / PI_WIDTH),
                    int(desired_point[1] * ARDUINO_RANGE / PI_HEIGHT)
                )

                # Prepare I2C data
                data = []
                for val in [scaled_current[0], scaled_current[1],
                            scaled_desired[0], scaled_desired[1]]:
                    data.extend(int16_to_bytes(val))
                
                bus.write_i2c_block_data(arduino_address, 0, data)
                last_send_time = now
                
            except Exception as e:
                print(f"I2C Error: {e}")

    # ===== Visualization =====
    # Draw tracking ROI
    if tracking_roi:
        x, y, w, h = tracking_roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    # Draw position history
    y_offset = 30
    for idx, offset in enumerate([2, 4, 6, 8, 10]):
        if len(position_history) >= offset:
            pos = position_history[-offset]
            color = (0, 255 - idx*50, idx*50)
            cv2.circle(frame, pos, 7-idx, color, -1)
            cv2.putText(frame, f"T-{offset}: {pos}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
    
    # Draw path and targets
    if show_path:
        for vx, vy in visited_positions:
            cv2.circle(frame, (vx, vy), 1, (255, 255, 255), -1)
    
    for idx, (px, py) in enumerate(intersection_points):
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
        cv2.putText(frame, str(idx+1), (px+5, py-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    if current_pos and desired_point:
        cv2.arrowedLine(frame, current_pos, desired_point, 
                        (255, 0, 255), 3, tipLength=0.3)
        cv2.putText(frame, f"Target: {desired_point}", 
                    (desired_point[0]+15, desired_point[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    cv2.imshow("Tracking", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_path = not show_path
    elif key == ord('c'):
        visited_positions.clear()

video_capture.release()
cv2.destroyAllWindows()
