import cv2
import numpy as np

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
last_ball_position = None
current_ball_position = None
intersection_points = []

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
            
            # Detect edge intersections
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
                    
                    # Update ball positions
                    if current_ball_position:
                        last_ball_position = current_ball_position
                    current_ball_position = (center_x, center_y)
                    
                    # Draw tracking elements
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.rectangle(frame, (x+x_local, y+y_local), 
                                (x+x_local+w_local, y+y_local+h_local), (0,255,0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (255,0,0), -1)
                    detected = True
                    
                    # Detect intersections
                    intersection_points = detect_black_curve(roi, (x, y, w, h))
                    break

    if not detected:
        tracking_roi = None
        current_ball_position = None
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

    # Draw information overlay
    y_offset = 30
    line_height = 25
    
    # Ball positions
    cv2.putText(frame, f"Current Ball: {current_ball_position}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += line_height
    
    cv2.putText(frame, f"Last Ball: {last_ball_position}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += line_height
    
    # Intersection points
    cv2.putText(frame, f"Intersections ({len(intersection_points)}):", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    y_offset += line_height
    
    for idx, (px, py) in enumerate(intersection_points):
        if y_offset > frame_height - 30: break
        cv2.putText(frame, f"Point {idx+1}: ({px}, {py})", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
        y_offset += line_height - 5

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
