import cv2
import numpy as np
import smbus

# ========== I2C 初始化 ==========
bus = smbus.SMBus(1)
arduino_address = 0x08
PATH_SCALE_FACTOR = 0.8
LOOK_AHEAD_POINTS = 40  # Number of points to look ahead on the path
MIN_PATH_POINTS = 5     # Minimum points needed for path following

def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val
    high = (val >> 8) & 0xFF
    low = val & 0xFF
    return [high, low]

def send_two_points_16bit(x1, y1, x2, y2):
    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}), ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")

def find_target_point(current_pos, path_points, look_ahead):
    if len(path_points) < MIN_PATH_POINTS:
        return None
    
    # Calculate distances to all path points
    distances = np.linalg.norm(path_points - current_pos, axis=1)
    
    # Find nearest point index
    nearest_idx = np.argmin(distances)
    
    # Calculate target index with look ahead
    target_idx = min(nearest_idx + look_ahead, len(path_points) - 1)
    
    return tuple(path_points[target_idx].astype(int))
'''
# Define HSV color ranges for red
lower_red_1 = np.array([0, 150, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 150, 100])
upper_red_2 = np.array([180, 255, 255])
'''
lower_red_1 = np.array([0, 100, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 50])
upper_red_2 = np.array([180, 255, 255])
# Initialize camera

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)
'''
# ========== 摄像头设置 / Camera Setup ==========
# 使用GStreamer管道强制指定MJPEG格式和分辨率/Use GStreamer pipeline for MJPEG format
pipeline = (
    "v4l2src device=/dev/video0 ! "                # 摄像头源/Camera source
    "image/jpeg,width=1280,height=720,framerate=60/1 ! "  # MJPEG格式和分辨率/MJPEG format & resolution
    "jpegdec ! "                                    # JPEG解码/JPEG decoding
    "videoconvert ! "                               # 颜色空间转换/Color conversion
    "appsink"                                       # 输出到OpenCV/Output to OpenCV
)

# 打开摄像头/Open camera
video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# 检查摄像头是否成功打开/Check if camera opened successfully
if not video_capture.isOpened():
    print("错误：无法打开摄像头/Error: Failed to open camera")
    exit()
'''
# Get frame dimensions
ret, test_frame = video_capture.read()
if ret:
    FRAME_HEIGHT, FRAME_WIDTH = test_frame.shape[:2]
else:
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

# Load and scale path coordinates
path_points = np.array([])
try:
    # Load coordinates as float for scaling
    path_points = np.loadtxt('maze_path_coordinates.csv', delimiter=',', dtype=np.float32)
    
    if len(path_points) > 0:
        # Get bounding box of original path
        min_x, min_y = np.min(path_points, axis=0)
        max_x, max_y = np.max(path_points, axis=0)
        path_width = max_x - min_x
        path_height = max_y - min_y
    
        # Calculate scaling to fit within scaled portion of frame
        max_allowed_width = FRAME_WIDTH * PATH_SCALE_FACTOR
        max_allowed_height = FRAME_HEIGHT * PATH_SCALE_FACTOR
    
        # Maintain aspect ratio
        scale = min(max_allowed_width/path_width, max_allowed_height/path_height)
    
        # Apply scaling
        path_points[:, 0] = (path_points[:, 0] - min_x) * scale
        path_points[:, 1] = (path_points[:, 1] - min_y) * scale
    
        # Center the path
        scaled_width = path_width * scale
        scaled_height = path_height * scale
        path_points[:, 0] += (FRAME_WIDTH - scaled_width) // 2
        path_points[:, 1] += (FRAME_HEIGHT - scaled_height) // 2
        
        # Convert to integer coordinates for drawing
        path_points = path_points.astype(np.int32)

except Exception as e:
    print(f"Error loading/scaling path: {e}")

tracking_roi = None  # (x, y, w, h)
TRACKING_SIZE = 100   # Size of the tracking box

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    full_red_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    detected = False
    target_point = None

    if tracking_roi is not None:
        # Extract ROI coordinates and ensure they're within frame bounds
        x, y, w, h = tracking_roi
        x, y = max(0, x), max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)
        
        if w > 0 and h > 0:
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create masks for red detection within ROI
            mask1 = cv2.inRange(hsv_roi, lower_red_1, upper_red_1)
            mask2 = cv2.inRange(hsv_roi, lower_red_2, upper_red_2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Morphological operations
            kernel = np.ones((5,5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)
            
            # Update full red mask for visualization
            full_red_mask[y:y+h, x:x+w] = red_mask
            
            # Find contours in ROI
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    (x_local, y_local, w_local, h_local) = cv2.boundingRect(contour)
                    # Convert to global coordinates
                    center_x = x + x_local + w_local//2
                    center_y = y + y_local + h_local//2
                    
                    # Update tracking ROI position
                    new_x = center_x - TRACKING_SIZE//2
                    new_y = center_y - TRACKING_SIZE//2
                    # Clamp to frame dimensions
                    new_x = max(0, min(new_x, frame_width - TRACKING_SIZE))
                    new_y = max(0, min(new_y, frame_height - TRACKING_SIZE))
                    tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                    
                    # Find target point on path
                    current_pos = np.array([center_x, center_y])
                    target_point = find_target_point(current_pos, path_points, LOOK_AHEAD_POINTS)
                    
                    if target_point is not None:
                        target_x, target_y = target_point
                        #send_two_points_16bit(center_x, center_y, target_x, target_y)
                        #send_two_points_16bit(center_x, center_y, target_x, target_y)
                        send_two_points_16bit(target_x, target_y, center_x, center_y)
                        
                        # Draw target point and connection line
                        cv2.circle(frame, (target_x, target_y), 8, (0,0,255), -1)
                        cv2.line(frame, (center_x, center_y), (target_x, target_y), (255,0,255), 2)

                    # Draw tracking elements
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Tracking box
                    cv2.rectangle(frame, (x+x_local, y+y_local), 
                                  (x+x_local+w_local, y+y_local+h_local), (0,255,0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (255,0,0), -1)
                    cv2.putText(frame, f"({center_x}, {center_y})", 
                                (center_x+10, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    detected = True
                    break

    if not detected:
        # Process full frame if not tracking or lost target
        tracking_roi = None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)
        full_red_mask = red_mask
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                (x, y, w, h) = cv2.boundingRect(contour)
                center_x = x + w//2
                center_y = y + h//2
                # Initialize tracking ROI
                new_x = center_x - TRACKING_SIZE//2
                new_y = center_y - TRACKING_SIZE//2
                new_x = max(0, min(new_x, frame_width - TRACKING_SIZE))
                new_y = max(0, min(new_y, frame_height - TRACKING_SIZE))
                tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                
                # Find target point on path
                current_pos = np.array([center_x, center_y])
                target_point = find_target_point(current_pos, path_points, LOOK_AHEAD_POINTS)
                
                if target_point is not None:
                    target_x, target_y = target_point
                    #send_two_points_16bit(center_x, center_y, target_x, target_y)
                    send_two_points_16bit(target_x, target_y, center_x, center_y)
                    
                    # Draw target point and connection line
                    cv2.circle(frame, (target_x, target_y), 8, (0,0,255), -1)
                    cv2.line(frame, (center_x, center_y), (target_x, target_y), (255,0,255), 2)

                # Draw elements
                cv2.rectangle(frame, (new_x, new_y), 
                             (new_x+TRACKING_SIZE, new_y+TRACKING_SIZE), (0,255,255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (255,0,0), -1)
                cv2.putText(frame, f"({center_x}, {center_y})", 
                            (center_x+10, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                break

    # Draw the predefined path from CSV
    if len(path_points) >= 2:
        pts = path_points.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

    # Display results
    cv2.imshow("Red Mask", full_red_mask)
    cv2.imshow("Tracking Red Object", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
