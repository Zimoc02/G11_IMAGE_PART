import cv2
import numpy as np
import math
import smbus
from collections import deque
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import time  # 加在最顶部和其他 import 放一起
import csv

# I2C 设置
bus = smbus.SMBus(1)
arduino_address = 0x08

def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val
    return [(val >> 8) & 0xFF, val & 0xFF]
'''
def send_two_points_16bit(x1, y1, x2, y2):
    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}) -> ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")
'''
def send_two_points_16bit(x1, y1, x2, y2):
    global last_send_time
    now = time.time()
    if now - last_send_time < SEND_INTERVAL:
        print(f"time out")
        return  # 时间间隔不足，不发送
    last_send_time = now  # 更新时间戳

    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}) -> ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")

# 参数
INTERPOLATION_COUNT = 2
headidx = 30
lower_red_1 = np.array([0, 100, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 50])
upper_red_2 = np.array([180, 255, 255])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
last_send_time = 0  # 全局变量，记录上次发送的时间戳
SEND_INTERVAL = 0.04  # 发送间隔 50 毫秒
errors = []  # 用于保存每帧的误差值
timestamps = []  # 保存每帧的时间戳
history_points = deque(maxlen=300)  # 红球历史坐标，用于画轨迹

x__1 = 0
x__2 = 0
y__1 = 0
y__2 = 0

# 实时识别红球位置
def detect_red_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red_1, upper_red_1),
        cv2.inRange(hsv, lower_red_2, upper_red_2)
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_center = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100 and area > max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            red_center = (y + h // 2, x + w // 2)
            max_area = area
    return red_center

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

# 路径图层函数
def generate_path_overlay(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    selected_path_image = np.zeros_like(image)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # === 洞口过滤逻辑 ===
        if area > 2000:  # 跳过太大的区域，避免识别到洞
            continue
        if cv2.isContourConvex(cnt):  # 跳过闭合环
            continue

        if 50 < area < 2000 and perimeter > 80:
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) > 5:
                cv2.drawContours(selected_path_image, [cnt], -1, (255, 255, 255), 2)

    # 骨架提取
    gray_selected = cv2.cvtColor(selected_path_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_selected, 127, 255, cv2.THRESH_BINARY)
    skeleton = cv2.ximgproc.thinning(binary)
    non_zero_points = np.column_stack(np.where(skeleton > 0))

    # 红球检测
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red_1, upper_red_1),
        cv2.inRange(hsv, lower_red_2, upper_red_2)
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_center = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100 and area > max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            red_center = (y + h // 2, x + w // 2)
            max_area = area

    # fallback 起点：若未识别红球，则使用图像中心最近的骨架点
    if red_center is None:
        print("⚠️ 未检测到红球，自动使用路径骨架上的点作为起点")
        if len(non_zero_points) == 0:
            return image.copy(), None, []
        center_yx = np.array([image.shape[0] // 2, image.shape[1] // 2])
        distances = np.linalg.norm(non_zero_points - center_yx, axis=1)
        nearest_idx = np.argmin(distances)
        red_center = tuple(non_zero_points[nearest_idx])

    # 稀疏化骨架点
    kdtree = cKDTree(non_zero_points)
    placed_points, used_indices = [], set()
    for idx, point in enumerate(non_zero_points):
        if idx not in used_indices:
            placed_points.append(point)
            used_indices.update(kdtree.query_ball_point(point, r=15))
    placed_points = np.array(placed_points)

    # 从红球起点出发构建有序路径
    all_points = placed_points.tolist()
    ordered_path = [red_center]
    used = set()
    current_point = red_center
    kdtree_path = cKDTree(all_points)

    while True:
        distances, indices = kdtree_path.query(current_point, k=len(all_points))
        next_point = None
        for idx in indices:
            candidate = tuple(all_points[idx])
            if candidate not in used and calculate_distance(current_point, candidate) < 65:
                next_point = candidate
                break
        if next_point is None:
            break
        ordered_path.append(next_point)
        used.add(next_point)
        current_point = next_point

    # 插值生成 refined_path
    refined_path = []
    for i in range(len(ordered_path) - 1):
        p1 = np.array(ordered_path[i])
        p2 = np.array(ordered_path[i + 1])
        refined_path.append(tuple(p1))
        for j in range(1, INTERPOLATION_COUNT + 1):
            ratio = j / (INTERPOLATION_COUNT + 1)
            new_point = tuple(((1 - ratio) * p1 + ratio * p2).astype(int))
            refined_path.append(new_point)
    refined_path.append(tuple(ordered_path[-1]))

    # ✅ ✅ ✅ 在这里插入 spline 平滑部分 ✅ ✅ ✅
    if len(refined_path) >= 4:
        try:
            path_array = np.array(refined_path)
            y_points, x_points = path_array[:, 0], path_array[:, 1]

            from scipy.interpolate import splprep, splev
            tck, u = splprep([x_points, y_points], s=1000, k=3)
            u_fine = np.linspace(0, 1, len(refined_path) * 5)
            x_smooth, y_smooth = splev(u_fine, tck)
            refined_path = [(int(y), int(x)) for x, y in zip(x_smooth, y_smooth)]
        except Exception as e:
            print(f"⚠️ B样条拟合失败，保留原始插值路径。原因: {e}")
    else:
        print("⚠️ 路径点数不足，无法执行 B样条平滑")

    # ✅ 保留路径可视化
    overlay = image.copy()
    for i in range(len(refined_path) - 1):
        cv2.line(overlay, (refined_path[i][1], refined_path[i][0]),
                      (refined_path[i + 1][1], refined_path[i + 1][0]), (0, 255, 255), 1)

    return overlay, red_center, refined_path


# 摄像头启动
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=60/1 ! "
    "jpegdec ! videoconvert ! appsink"
)
video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not video_capture.isOpened():
    print("错误：无法打开摄像头")
    exit()

ret, initial_frame = video_capture.read()
if not ret:
    print("错误：无法获取初始图像")
    exit()

path_overlay, red_center, refined_path = generate_path_overlay(initial_frame)
print(f"路径起点：{red_center}，路径长度：{len(refined_path)}")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    red_center = detect_red_ball(frame)
    display = cv2.addWeighted(path_overlay, 0.6, frame, 0.4, 0)

    if red_center:
        # 当前红球位置：红色圆圈
        cv2.circle(display, (red_center[1], red_center[0]), 8, (0, 0, 255), -1)
        cv2.putText(display, "Current", (red_center[1] + 10, red_center[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 查找 refined_path 中最接近 red_center 的点，然后+3个点
        if refined_path:
            distances = [calculate_distance(red_center, pt) for pt in refined_path]
            nearest_idx = int(np.argmin(distances))
            target_idx = nearest_idx + headidx
            if target_idx < len(refined_path):
                target_point = refined_path[target_idx]

                # 目标点：紫色圆圈
                cv2.circle(display, (target_point[1], target_point[0]), 8, (255, 0, 255), -1)
                cv2.putText(display, "Target", (target_point[1] + 10, target_point[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # 画箭头：从当前红球指向目标点（紫色箭头）
                cv2.arrowedLine(display,
                                (red_center[1], red_center[0]),
                                (target_point[1], target_point[0]),
                                (255, 0, 255), 2, tipLength=0.3)
                '''                
                for i, pt in enumerate(history_points):
                    if i >= len(errors):
                        break
                    err_val = errors[i]
                    err_clamped = min(int(err_val * 3), 255)  # 控制颜色范围
                    color = (0, 255 - err_clamped, err_clamped)  # 绿色 = 精确，红色 = 偏差大
                    cv2.circle(display, (pt[1], pt[0]), 2, color, -1)
                '''
                
                nearest_path_point = refined_path[nearest_idx]
                euclidean_error = calculate_distance(red_center, nearest_path_point)

                # 保存误差与时间
                errors.append(euclidean_error)
                timestamps.append(time.time())
                history_points.append(red_center)  # 添加到历史轨迹中

                # 每 30 帧打印统计
                if len(errors) % 30 == 0:
                    print(f"[误差统计] 平均: {np.mean(errors):.2f} px, 最大: {np.max(errors):.2f} px, 标准差: {np.std(errors):.2f} px")

                send_two_points_16bit(red_center[1], red_center[0], target_point[1], target_point[0])
                
                x__1 = red_center[1]
                x__2 = target_point[1]
                y__1 = red_center[0]
                y__2 = target_point[0]
                
    cv2.imshow("Red Ball Tracking", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        ret, new_frame = video_capture.read()
        if ret:
            path_overlay, red_center, refined_path = generate_path_overlay(new_frame)
            print("[热键P] 重新生成路径图层")
    elif key == ord('t'):
        if red_center and refined_path:
            distances = [calculate_distance(red_center, pt) for pt in refined_path]
            nearest_idx = int(np.argmin(distances))
            target_idx = nearest_idx + headidx
            if target_idx < len(refined_path):
                target_point = refined_path[target_idx]
                print(f"[热键T] 当前红球位置: {red_center}，目标点: {target_point}")
            else:
                print("[热键T] 红球靠近路径尾部，无法获取目标点")
    elif key == ord('s'):
    with open('tracking_accuracy.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Error(px)'])
        for t, e in zip(timestamps, errors):
            writer.writerow([t, e])
    print("✅ [热键S] 已保存误差数据到 tracking_accuracy.csv")

video_capture.release()
cv2.destroyAllWindows()
