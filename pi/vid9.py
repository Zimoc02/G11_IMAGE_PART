import cv2
import numpy as np
from collections import deque





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
position_history = deque(maxlen=2)  # Stores last two positions
intersection_points = []

def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

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
    farthest_points = []

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
                    if len(position_history) == 0 or position_history[-1] != (center_x, center_y):
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

    # Find farthest points for last two positions
    if len(position_history) >= 1 and len(intersection_points) > 0:
        # For current position
        current_distances = [calculate_distance(pos, position_history[-1]) 
                           for pos in intersection_points]
        if current_distances:
            farthest_current = intersection_points[np.argmax(current_distances)]
            farthest_points.append(farthest_current)
        
        # For previous position (if exists)
        if len(position_history) >= 2:
            prev_distances = [calculate_distance(pos, position_history[-2]) 
                            for pos in intersection_points]
            if prev_distances:
                farthest_prev = intersection_points[np.argmax(prev_distances)]
                farthest_points.append(farthest_prev)

    # Draw information
    y_offset = 30
    line_height = 25
    
    # Position history
    for idx, pos in enumerate(reversed(position_history)):
        if idx >= 2: break
        cv2.putText(frame, f"Ball Position [-{idx}]: {pos}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
    
    # Intersection points
    cv2.putText(frame, f"Intersections ({len(intersection_points)}):", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    y_offset += line_height
    
    # Draw all intersection points in yellow
    for px, py in intersection_points:
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
    
    # Draw farthest points in pink
    for point in farthest_points:
        px, py = point
        cv2.circle(frame, (px, py), 8, (255, 192, 203), -1)  # Pink color
        cv2.putText(frame, f"Farthest ({px}, {py})", (px+10, py-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
