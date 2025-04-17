import cv2
import numpy as np
from collections import deque
import math
import smbus
import json
from scipy.spatial import KDTree

# ========== I2C 初始化 / I2C Initialization ==========
bus = smbus.SMBus(1)
arduino_address = 0x08  # Arduino I2C 地址 / I2C address of the Arduino

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

# ========== 颜色范围 / Color Ranges ==========
lower_red_1 = np.array([0, 100, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 50])
upper_red_2 = np.array([180, 255, 255])

# ========== 路径数据 / Path Data ==========
# Generate this using the path extraction code and save as JSON
# ========== 路径数据 / Path Data ==========
with open("path_data.json") as f:
    PATH_DATA = json.load(f)
dense_path = np.array(PATH_DATA)
kdtree = KDTree(dense_path)
last_index = 0
window_size = 150  # 搜索窗口大小 / Search window size

# ========== 摄像头设置 / Camera Setup ==========
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=60/1 ! "
    "jpegdec ! "
    "videoconvert ! "
    "appsink"
)
video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not video_capture.isOpened():
    print("错误：无法打开摄像头/Error: Failed to open camera")
    exit()
# ========== Add these calibration parameters ==========
CALIB_FILE = "calibration.json"
# Original image size used for path extraction
PATH_IMG_SIZE = (3735, 2957)  # Replace with your path image dimensions
# Camera resolution
CAMERA_SIZE = (1280, 720)

# Scaling factors (initialize with default values)
scale_x = CAMERA_SIZE[0] / PATH_IMG_SIZE[0]
scale_y = CAMERA_SIZE[1] / PATH_IMG_SIZE[1]
offset_x = 0
offset_y = 0

# ========== Add calibration functions ==========
def load_calibration():
    global scale_x, scale_y, offset_x, offset_y
    try:
        with open(CALIB_FILE, 'r') as f:
            data = json.load(f)
            scale_x = data['scale_x']
            scale_y = data['scale_y']
            offset_x = data['offset_x']
            offset_y = data['offset_y']
        print("Loaded calibration data")
    except FileNotFoundError:
        print("Using default calibration values")

def save_calibration():
    data = {
        'scale_x': scale_x,
        'scale_y': scale_y,
        'offset_x': offset_x,
        'offset_y': offset_y
    }
    with open(CALIB_FILE, 'w') as f:
        json.dump(data, f)
    print("Calibration data saved")

def scale_coordinates(point):
    x = point[0] * scale_x + offset_x
    y = point[1] * scale_y + offset_y
    return (int(x), int(y))


# ========== 跟踪参数 / Tracking Parameters ==========
TRACKING_RADIUS = 65
position_history = deque(maxlen=10)
visited_positions = set()
show_path = True

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def get_desired_position(current_pos):
    global last_index
    start_idx = max(0, last_index - window_size)
    end_idx = min(len(dense_path), last_index + window_size)
    search_subset = dense_path[start_idx:end_idx]
    
    if search_subset.size == 0:
        return None
    
    distances = np.linalg.norm(search_subset - current_pos, axis=1)
    nearest_idx = np.argmin(distances)
    last_index = start_idx + nearest_idx
    return dense_path[last_index]

# ========== 主循环 / Main Loop ==========
tracking_center = None
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    detected = False
    desired_pos = None

    # ======= 红球追踪 / Ball Tracking =======
    if tracking_center is not None:
        cx, cy = tracking_center
        radius = TRACKING_RADIUS

        # ROI处理 / ROI Handling
        x_start = max(0, cx - radius)
        y_start = max(0, cy - radius)
        x_end = min(frame_width, cx + radius)
        y_end = min(frame_height, cy + radius)
        roi_square = frame[y_start:y_end, x_start:x_end]

        if roi_square.size == 0:
            tracking_center = None
            continue

        # 红球检测 / Ball Detection
        hsv_roi = cv2.cvtColor(roi_square, cv2.COLOR_BGR2HSV)
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
                global_x = x_start + x_local + w_local // 2
                global_y = y_start + y_local + h_local // 2
                tracking_center = (global_x, global_y)
                position_history.append((global_x, global_y))
                detected = True

                # 更新路径记录 / Update path history
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        visited_positions.add((global_x + dx, global_y + dy))
                break

    if not detected:
        tracking_center = None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_frame, lower_red_1, upper_red_1),
            cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        )
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 20:
                (x, y, w, h) = cv2.boundingRect(contour)
                tracking_center = (x + w//2, y + h//2)
                break

    # ======= 路径跟随逻辑 / Path Following Logic =======
    if tracking_center and position_history:
        current_pos = np.array(tracking_center)
        desired_pos = get_desired_position(current_pos)

    # ======= 发送至Arduino / Send to Arduino =======
    if desired_pos is not None and position_history:
        current_pos = position_history[-1]
        send_two_points_16bit(current_pos[0], current_pos[1], 
                            int(desired_pos[0]), int(desired_pos[1]))

    # ======= 可视化 / Visualization =======
     # ======= 可视化 / Visualization =======
# ========== Modify the visualization section ==========
# Change the path drawing to:

    if show_path:
        sample_step = 5
        for point in dense_path[::sample_step]:
            scaled_point = scale_coordinates(point)
            cv2.circle(frame, scaled_point, 2, (255, 255, 0), -1)

    if tracking_center:
        cx, cy = tracking_center
        cv2.circle(frame, (cx, cy), TRACKING_RADIUS, (0, 255, 255), 2)

    if desired_pos is not None:
        cv2.circle(frame, tuple(desired_pos.astype(int)), 8, (0, 0, 255), -1)
        if position_history:
            current_pos = position_history[-1]
            cv2.arrowedLine(frame, tuple(current_pos), tuple(desired_pos.astype(int)), 
                          (255, 0, 0), 2, tipLength=0.3)

    # Existing visited path drawing
    if show_path:
        for vx, vy in visited_positions:
            cv2.circle(frame, (vx, vy), 1, (255, 255, 255), -1)

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
