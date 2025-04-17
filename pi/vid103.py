import cv2
import numpy as np
from collections import deque
import math
import smbus

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
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

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

# ========== 跟踪参数 / Tracking Parameters ==========
TRACKING_RADIUS = 53
CURVE_PARAMS = {
    'min_contour_area': 50,
    'approx_epsilon': 0.02,
    'direction_samples': 5
}

# ========== 跟踪状态变量 / Tracking State Variables ==========
tracking_center = None  # (cx, cy)
position_history = deque(maxlen=10)
intersection_points = []
farthest_point = None
prediction_angle = None
visited_positions = set()
show_path = True

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def detect_black_curve(roi_square, roi_position, mask):
    x_start, y_start, w, h = roi_position
    hsv = cv2.cvtColor(roi_square, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_mask = cv2.bitwise_and(black_mask, mask)
    kernel = np.ones((5, 5), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    current_intersections = []
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > CURVE_PARAMS['min_contour_area']:
            perimeter = cv2.arcLength(contour, True)
            epsilon = CURVE_PARAMS['approx_epsilon'] * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            for point in approx:
                px, py = point[0]
                if px == 0 or px == w - 1 or py == 0 or py == h - 1:
                    global_x = x_start + px
                    global_y = y_start + py
                    current_intersections.append((global_x, global_y))

    return current_intersections, black_mask

# ========== 主循环 / Main Loop ==========
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    detected = False
    intersection_points = []
    farthest_point = None
    prediction_angle = None

    # ======= 如果已有追踪中心 / If tracking center exists =======
    if tracking_center is not None:
        cx, cy = tracking_center
        radius = TRACKING_RADIUS

        # 提取圆形ROI周围的方形区域 / Extract square region around the circle
        x_start = max(0, cx - radius)
        y_start = max(0, cy - radius)
        x_end = min(frame_width, cx + radius)
        y_end = min(frame_height, cy + radius)
        roi_square = frame[y_start:y_end, x_start:x_end]

        if roi_square.size == 0:
            tracking_center = None
            continue

        # 创建圆形掩模 / Create circular mask
        mask = np.zeros(roi_square.shape[:2], dtype=np.uint8)
        center_x_in_roi = cx - x_start
        center_y_in_roi = cy - y_start
        cv2.circle(mask, (center_x_in_roi, center_y_in_roi), radius, 255, -1)

        # ======= 红球检测 / Detect red ball =======
        hsv_roi = cv2.cvtColor(roi_square, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_roi, lower_red_1, upper_red_1),
            cv2.inRange(hsv_roi, lower_red_2, upper_red_2)
        )
        red_mask = cv2.bitwise_and(red_mask, mask)
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 200:
                (x_local, y_local, w_local, h_local) = cv2.boundingRect(contour)
                # 计算全局坐标 / Calculate global coordinates
                global_x = x_start + x_local + w_local // 2
                global_y = y_start + y_local + h_local // 2
                tracking_center = (global_x, global_y)
                position_history.append((global_x, global_y))
                cv2.putText(frame, f"Ball:({global_x},{global_y})",
                            (global_x+10, global_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                detected = True

                # 更新访问路径 / Update visited path
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        vx = global_x + dx
                        vy = global_y + dy
                        if 0 <= vx < frame_width and 0 <= vy < frame_height:
                            visited_positions.add((vx, vy))
                # 清理旧路径点 / Clean old path points
                erase_radius = 6* TRACKING_RADIUS
                new_visited = set()
                for vx, vy in visited_positions:
                    if calculate_distance((vx, vy), (global_x, global_y)) <= erase_radius:
                        new_visited.add((vx, vy))
                visited_positions = new_visited

                # 检测黑线交点 / Detect black curve intersections
                intersection_points, black_mask = detect_black_curve(
                    roi_square, (x_start, y_start, roi_square.shape[1], roi_square.shape[0]), mask
                )
                colored_mask = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
                frame[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                    frame[y_start:y_end, x_start:x_end], 0.7, colored_mask, 0.3, 0)
                break

    # ====== 如果未检测到红球 / If red ball not detected ======
    if not detected:
        tracking_center = None
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
                center_x = x + w // 2
                center_y = y + h // 2
                tracking_center = (center_x, center_y)
                break

    # ====== 智能选择交点 / Smart intersection selection ======
    if visited_positions and intersection_points and tracking_center:
        cx, cy = tracking_center
        filtered_visited = [(vx, vy) for vx, vy in visited_positions
                            if not (cx - TRACKING_RADIUS <= vx <= cx + TRACKING_RADIUS and
                                    cy - TRACKING_RADIUS <= vy <= cy + TRACKING_RADIUS)]

        for (fx, fy) in filtered_visited:
            cv2.circle(frame, (fx, fy), 2, (144, 238, 144), -1)

        if filtered_visited:
            total = len(filtered_visited)
            best_score = -1
            for ip in intersection_points:
                score = 0
                for idx, vp in enumerate(filtered_visited):
                    weight = (idx + 1) / total
                    dist = calculate_distance(ip, vp)
                    score += weight * dist
                if score > best_score:
                    best_score = score
                    farthest_point = ip

    # ====== 方向预测与发送 / Predict angle and send over I2C ======
    if farthest_point and position_history:
        current_pos = position_history[-1]
        dx = farthest_point[0] - current_pos[0]
        dy = farthest_point[1] - current_pos[1]
        prediction_angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
        send_two_points_16bit(current_pos[0], current_pos[1], farthest_point[0], farthest_point[1])

    # ====== 可视化 / Visualization ======
    y_offset = 30
    line_height = 25

    if tracking_center:
        cx, cy = tracking_center
        cv2.circle(frame, (cx, cy), TRACKING_RADIUS, (0, 255, 255), 2)

    for idx, offset in enumerate([2, 4, 6, 8, 10]):
        if len(position_history) >= offset:
            pos = position_history[-offset]
            color = (0, 255 - idx * 50, idx * 50)
            cv2.circle(frame, pos, 7 - idx, color, -1)
            cv2.putText(frame, f"T-{offset}: {pos}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height

    for idx, (px, py) in enumerate(intersection_points):
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"{idx+1}", (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if farthest_point and prediction_angle is not None:
        cv2.circle(frame, farthest_point, 10, (255, 192, 203), -1)
        if position_history:
            current_pos = position_history[-1]
            cv2.arrowedLine(frame, current_pos, farthest_point, (255, 192, 203), 2, tipLength=0.3)
            cv2.putText(frame, f"Angle: {prediction_angle:.1f}°",
                        (farthest_point[0] + 15, farthest_point[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 192, 203), 2)

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
