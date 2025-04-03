
import cv2
import numpy as np
from collections import deque
import math
import smbus

# ========== I2C 初始化 / I2C Initialization ==========
bus = smbus.SMBus(1)
arduino_address = 0x08  # Arduino I2C 地址 / I2C address of the Arduino

# 整数转换为 16 位高低字节 / Convert int to two 8-bit bytes (for I2C)
def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val  # 处理负数 / Handle negative values
    high = (val >> 8) & 0xFF
    low = val & 0xFF
    return [high, low]

# 发送两个点（x1,y1）和（x2,y2）/ Send two 16-bit points via I2C
def send_two_points_16bit(x1, y1, x2, y2):
    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}), ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")
        print(f"po: ({x1}, {y1}), ({x2}, {y2})")

# ========== 颜色范围 / Color Ranges ==========
lower_red_1 = np.array([0, 100, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 50])
upper_red_2 = np.array([180, 255, 255])
#black
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
# ========== 摄像头设置 / Camera Setup ==========
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
# ========== 跟踪参数 / Tracking Parameters ==========
TRACKING_SIZE =100
CURVE_PARAMS = {
    'min_contour_area': 50,
    'approx_epsilon': 0.02,
    'direction_samples': 5
}

# ========== 跟踪状态变量 / Tracking State Variables ==========
tracking_roi = None
position_history = deque(maxlen=10)
intersection_points = []
farthest_point = None
prediction_angle = None
visited_positions = set()
show_path = True

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def detect_black_curve(roi, roi_position):
    x, y, w, h = roi_position
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
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
                if px == 0 or px == w - 1 or py == 0 or py == h - 1:
                    global_x = x + px
                    global_y = y + py
                    current_intersections.append((global_x, global_y))

    return current_intersections, mask
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

    # ======= 如果已有追踪区域 / If tracking ROI exists =======
    if tracking_roi is not None:
        x, y, w, h = tracking_roi
        x, y = max(0, x), max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)

        if w > 0 and h > 0:
            roi = frame[y:y + h, x:x + w]

            # ======= 红球检测 / Detect red ball =======
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv_roi, lower_red_1, upper_red_1),
                cv2.inRange(hsv_roi, lower_red_2, upper_red_2)
            )
            kernel = np.ones((5, 5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    (x_local, y_local, w_local, h_local) = cv2.boundingRect(contour)
                    center_x = x + x_local + w_local // 2
                    center_y = y + y_local + h_local // 2

                    new_x = max(0, min(center_x - TRACKING_SIZE // 2, frame_width - TRACKING_SIZE))
                    new_y = max(0, min(center_y - TRACKING_SIZE // 2, frame_height - TRACKING_SIZE))
                    tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)

                    position_history.append((center_x, center_y))
                    AUX_RADIUS = 3
                    AUX_POINTS_PER_CIRCLE = 12  # density

                    for i in range(AUX_POINTS_PER_CIRCLE):
                        angle = 2 * math.pi * i / AUX_POINTS_PER_CIRCLE
                        vx = int(center_x + AUX_RADIUS * math.cos(angle))
                        vy = int(center_y + AUX_RADIUS * math.sin(angle))
                        if 0 <= vx < frame_width and 0 <= vy < frame_height:
                            visited_positions.add((vx, vy))
                    cv2.putText(frame,f"Ball:({center_x},{center_y})",
                                (center_x+10, center_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2    
                                )
                    detected = True

                    # 灌水路径点
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            vx = center_x + dx
                            vy = center_y + dy
                            if 0 <= vx < frame_width and 0 <= vy < frame_height:
                                visited_positions.add((vx, vy))

                    # 清除远离当前点的旧路径点
                    erase_radius = 3.5 * TRACKING_SIZE
                    new_visited = set()
                    for vx, vy in visited_positions:
                        if calculate_distance((vx, vy), (center_x, center_y)) <= erase_radius:
                            new_visited.add((vx, vy))
                    visited_positions = new_visited

                    # 检测交点并可视化 mask
                    intersection_points, black_mask = detect_black_curve(roi, (x, y, w, h))
                    colored_mask = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
                    frame[y:y + h, x:x + w] = cv2.addWeighted(frame[y:y + h, x:x + w], 0.7, colored_mask, 0.3, 0)
                    break
        # ====== 如果未检测到红球 / If red ball not detected ======
    if not detected:
        tracking_roi = None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_frame, lower_red_1, upper_red_1),
            cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        )
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 20:
                (x, y, w, h) = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                new_x = max(0, min(center_x - TRACKING_SIZE // 2, frame_width - TRACKING_SIZE))
                new_y = max(0, min(center_y - TRACKING_SIZE // 2, frame_height - TRACKING_SIZE))
                tracking_roi = (new_x, new_y, TRACKING_SIZE, TRACKING_SIZE)
                break

    # ====== 智能选择交点 / Smart intersection selection ======
    if visited_positions and intersection_points and tracking_roi:
        visited_list = list(visited_positions)
        x_roi, y_roi, w_roi, h_roi = tracking_roi

        def is_outside_roi(px, py):
            return not (x_roi <= px <= x_roi + w_roi and y_roi <= py <= y_roi + h_roi)

        filtered_visited = [(vx, vy) for vx, vy in visited_list if is_outside_roi(vx, vy)]

        for (fx, fy) in filtered_visited:
            cv2.circle(frame, (fx, fy), 2, (144, 238, 144), -1)

        if filtered_visited:
            total = len(filtered_visited)
            best_score = -1
            # print("[Intersection Scoring]")
            for idx_ip, ip in enumerate(intersection_points):
                score = 0
                for idx, vp in enumerate(filtered_visited):
                    weight = (total-idx) / total
                    dist = calculate_distance(ip, vp)
                    score += weight * dist
                # print(f"  - Intersection {idx_ip + 1} at {ip}: Score = {score:.2f}")
                if score > best_score:
                    best_score = score
                    farthest_point = ip
            # print(f"  => Selected Intersection: {farthest_point} (Score: {best_score:.2f})")

    # ====== 方向预测与发送 / Predict angle and send over I2C ======
    if farthest_point and position_history:
        current_pos = position_history[-1]
        dx = farthest_point[0] - current_pos[0]
        dy = farthest_point[1] - current_pos[1]
        prediction_angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
        send_two_points_16bit(current_pos[0], current_pos[1], farthest_point[0], farthest_point[1])

    # ====== 可视化区域 / Visualization ======
    y_offset = 30
    line_height = 25

    if tracking_roi:
        x_roi, y_roi, w_roi, h_roi = tracking_roi
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 255), 2)

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

    # if position_history:
    #     print(f"[Ball] Current Position: {position_history[-1]}")
    # else:
    #     print("[Ball] Not detected")

    # if intersection_points:
    #     print(f"[Intersections] Detected {len(intersection_points)} point(s):")
    # else:
    #     print("[Intersections] None")

    # print("-" * 40)

    if show_path:
        for vx, vy in visited_positions:
            cv2.circle(frame, (vx, vy), 1, (255, 255, 255), -1)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_path = not show_path
        # print("[Hotkey] Toggled path display:", "ON" if show_path else "OFF")
    elif key == ord('c'):
        visited_positions.clear()
      
        if position_history:
            current_x, current_y = position_history[-1]
            for i in range(100):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 10)  # 控制半径范围，越大扩散越广
                dx = int(radius * np.cos(angle))
                dy = int(radius * np.sin(angle))
                vx = current_x + dx
                vy = current_y + dy
                if 0 <= vx < frame_width and 0 <= vy < frame_height:
                    visited_positions.add((vx, vy))
            print(f"[Hotkey] Added 100 historical points near ({current_x}, {current_y})")
        else:
            print("[Hotkey] No current position to add points.")
        # print("[Hotkey] Cleared visited path!")

video_capture.release()
cv2.destroyAllWindows()
