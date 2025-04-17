import cv2
import numpy as np
from collections import deque
import math

import smbus

bus = smbus.SMBus(1)  # I2C bus 1
arduino_address = 0x08  # Arduino I2C address

def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val
    high = (val >> 8) & 0xFF
    low = val & 0xFF
    return [high, low]  



def send_two_points_16bit(x1,y1,x2,y2):
    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}), ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")
        
# Color ranges
lower_red_1 = np.array([0, 150, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 150, 100])
upper_red_2 = np.array([180, 255, 255])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Initialize camera
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Tracking parameters
TRACKING_SIZE = 70
CURVE_PARAMS = {
    'min_contour_area': 50,
    'approx_epsilon': 0.02,
    'direction_samples': 5
}

# Tracking variables
tracking_roi = None
position_history = deque(maxlen=10)  # Stores last 10 positions
intersection_points = []
farthest_point = None
prediction_angle = None

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

def detect_black_curve(roi, roi_position):
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
                    global_x = x + px
                    global_y = y + py
                    current_intersections.append((global_x, global_y))

    return current_intersections

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    detected = False
    intersection_points = []
    farthest_point = None
    prediction_angle = None

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
            
            kernel = np.ones((5,5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    (x_local, y_local, w_local, h_local) = cv2.boundingRect(contour)
                    center_x = x + x_local + w_local//2
                    center_y = y + y_local + h_local//2
                    
                    new_x = center_x - TRACKING_SIZE//2
                    new_y = center_y - TRACKING_SIZE//2
                    new_x = max(0, min(new_x, frame_width - TRACKING_SIZE))
                    new_y = max(0, min(new_y, frame_height - TRACKING_SIZE))
                    tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                    
                    # Update position history
                    position_history.append((center_x, center_y))
                    detected = True
                    
                    # Detect intersections
                    intersection_points = detect_black_curve(roi, (x, y, w, h))
                    break

    if not detected:
        tracking_roi = None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_frame, lower_red_1, upper_red_1),
            cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        )
        
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 20:
                (x, y, w, h) = cv2.boundingRect(contour)
                center_x = x + w//2
                center_y = y + h//2
                
                new_x = center_x - TRACKING_SIZE//2
                new_y = center_y - TRACKING_SIZE//2
                new_x = max(0, min(new_x, frame_width - TRACKING_SIZE))
                new_y = max(0, min(new_y, frame_height - TRACKING_SIZE))
                tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                break

    # Find historical points at 2,4,6,8,10 steps back
    historical_points = []
    for offset in [2,4,6,8,10]:
        if len(position_history) >= offset:
            historical_points.append(position_history[-offset])

    # Find farthest intersection point
    if historical_points and intersection_points:
        max_distance = -1
        farthest_point = None
        
        for ip in intersection_points:
            for hp in historical_points:
                distance = calculate_distance(ip, hp)
                if distance > max_distance:
                    max_distance = distance
                    farthest_point = ip

    # Calculate prediction angle
    if farthest_point and position_history:
        current_pos = position_history[-1]
        dx = farthest_point[0] - current_pos[0]
        dy = farthest_point[1] - current_pos[1]
        prediction_angle = math.degrees(math.atan2(-dy, dx))
        send_two_points_16bit(current_pos[1],current_pos[0],farthest_point[1],farthest_point[0])
    # Visualization
    y_offset = 30
    line_height = 25

    # ✅ Draw ROI box if available (yellow rectangle)
    if tracking_roi is not None:
        x_roi, y_roi, w_roi, h_roi = tracking_roi
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 255), 2)  # Yellow rectangle for ROI box
		
		
    # Draw historical points
    for idx, offset in enumerate([2,3,4,5,6]):
        if len(position_history) >= offset:
            pos = position_history[-offset]
            color = (0, 255 - idx*50, idx*50)
            cv2.circle(frame, pos, 7 - idx, color, -1)
            cv2.putText(frame, f"T-{offset}: {pos}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height
    
    # Draw intersection points
    for px, py in intersection_points:
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
    
    # Draw prediction elements
    if farthest_point and prediction_angle is not None:
        cv2.circle(frame, farthest_point, 10, (255, 192, 203), -1)
        if position_history:
            current_pos = position_history[-1]
            cv2.arrowedLine(frame, current_pos, farthest_point, 
                           (255, 192, 203), 2, tipLength=0.3)
            cv2.putText(frame, f"Angle: {prediction_angle:.1f}°", 
                       (farthest_point[0] + 15, farthest_point[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 192, 203), 2)
    
    if position_history:
        print(f"[Ball] Current Position: {position_history[-1]}")
    else:
        print("[Ball] Not detected")

    # 打印所有粉色交点
    if intersection_points:
        print(f"[Intersections] Detected {len(intersection_points)} point(s):")
        for idx, (px, py) in enumerate(intersection_points):
            print(f"  - Point {idx+1}: ({px}, {py})")
    else:
        print("[Intersections] None")

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
video_capture.release()
cv2.destroyAllWindows()
