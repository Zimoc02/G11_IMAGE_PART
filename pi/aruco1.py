import cv2
import numpy as np
import math
import smbus
from collections import deque
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import time
from cv2 import aruco  # +++ Added ArUco import

# +++ ArUco Parameters 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)  # 4x4_50 dictionary
ARUCO_PARAMS = aruco.DetectorParameters_create()

# I2C Setup
bus = smbus.SMBus(1)
arduino_address = 0x08

def detect_board_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    
    sorted_corners = [None] * 4
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if 0 <= marker_id <= 3:
                pts = corners[i][0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                sorted_corners[marker_id] = (cx, cy)
    
    if None in sorted_corners:
        return None
    return np.array(sorted_corners, dtype=np.float32)


def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val
    return [(val >> 8) & 0xFF, val & 0xFF]

def send_two_points_16bit(x1, y1, x2, y2):
    global last_send_time
    now = time.time()
    if now - last_send_time < SEND_INTERVAL:
        print(f"time out")
        return
    last_send_time = now

    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}) -> ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")

# Parameters
INTERPOLATION_COUNT = 2
headidx = 30
lower_red_1 = np.array([0, 100, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 50])
upper_red_2 = np.array([180, 255, 255])
last_send_time = 0
SEND_INTERVAL = 0.04
errors = []
timestamps = []
history_points = deque(maxlen=300)
ref_pts = None
H_inv = None
roi_dict = {}
ROI_MARGIN = 80
aruco_roi_bounds = None  

# 下面是路径生成和主循环（保持你的原始逻辑即可）


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

def image_to_board_coords(point, H):
    """
    将图像坐标 point = (y, x) 转换为棋盘坐标系 (X, Y)
    """
    pt = np.array([[point[1], point[0], 1]], dtype=np.float32).T  # (x, y, 1)
    board_pt = H @ pt
    board_pt /= board_pt[2]
    return float(board_pt[0]), float(board_pt[1])  # 返回 (X, Y)

def detect_aruco_and_get_roi(frame, margin=100):
    corners, ids, _ = aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is not None:
        selected_corners = []
        for i, marker_id in enumerate(ids.flatten()):
            if 0 <= marker_id <= 3:
                selected_corners.append(corners[i])
        if len(selected_corners) == 4:
            all_corners = np.concatenate(selected_corners, axis=1)
            x_min = max(0, int(np.min(all_corners[:, 0, 0])) - margin)
            x_max = min(frame.shape[1], int(np.max(all_corners[:, 0, 0])) + margin)
            y_min = max(0, int(np.min(all_corners[:, 0, 1])) - margin)
            y_max = min(frame.shape[0], int(np.max(all_corners[:, 0, 1])) + margin)
            return (x_min, x_max, y_min, y_max), selected_corners, np.array([[0], [1], [2], [3]])
    return None, None, None

def detect_aruco_in_roi(frame, roi_bounds):
    x_min, x_max, y_min, y_max = roi_bounds
    roi = frame[y_min:y_max, x_min:x_max]
    corners, ids, _ = aruco.detectMarkers(roi, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is not None:
        for corner in corners:
            corner += np.array([[[x_min, y_min]]])  # 还原到全图坐标
    return corners, ids

# Camera Initialization
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not video_capture.isOpened():
    print("Error: Camera not opened")
    exit()

# +++ Initial Frame Marker Detection +++
ret, initial_frame = video_capture.read()
if not ret:
    print("Error: Initial frame not captured")
    exit()

gray_init = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
corners, ids, _ = aruco.detectMarkers(gray_init, ARUCO_DICT, parameters=ARUCO_PARAMS)
if ids is None or len(ids) < 4:
    print("❌ 初始帧未检测到足够的 ArUco markers (0~3)")
    exit()

# 计算四个角的中心点（根据 ID 0~3）
selected_corners = []
sorted_corners = [None] * 4
for i, marker_id in enumerate(ids.flatten()):
    if 0 <= marker_id <= 3:
        pts = corners[i][0]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        sorted_corners[marker_id] = (cx, cy)

if None in sorted_corners:
    print("❌ 四个角标识不全")
    exit()

initial_corners = np.array(sorted_corners, dtype=np.float32)

# 设置参考点，坐标系以中心为原点
width = (np.linalg.norm(initial_corners[1] - initial_corners[0]) +
         np.linalg.norm(initial_corners[2] - initial_corners[3])) / 2
height = (np.linalg.norm(initial_corners[3] - initial_corners[0]) +
          np.linalg.norm(initial_corners[2] - initial_corners[1])) / 2
ref_pts = np.array([
    [-width / 2, -height / 2],
    [ width / 2, -height / 2],
    [ width / 2,  height / 2],
    [-width / 2,  height / 2],
], dtype=np.float32)

H, _ = cv2.findHomography(initial_corners, ref_pts)
print("✅ Homography 初始化完成，坐标原点设为棋盘中心")
aruco_roi_bounds = None
if len(selected_corners) == 4:
    all_corners = np.concatenate(selected_corners, axis=1)
    margin = 80
    x_min = max(0, int(np.min(all_corners[:, 0, 0])) - margin)
    x_max = min(initial_frame.shape[1], int(np.max(all_corners[:, 0, 0])) + margin)
    y_min = max(0, int(np.min(all_corners[:, 0, 1])) - margin)
    y_max = min(initial_frame.shape[0], int(np.max(all_corners[:, 0, 1])) + margin)
    aruco_roi_bounds = (x_min, x_max, y_min, y_max)

print("✅ 初始 ArUco 四角检测完成，ROI 启用")
# Generate initial path
path_overlay, red_center, refined_path = generate_path_overlay(initial_frame)
print(f"Initial path length: {len(refined_path)}")

# Main Loop (Added Homography Calculation) +++
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    overlay, red_center, refined_path = generate_path_overlay(frame)

    # 🧠 红球位置转换到棋盘坐标系
    if red_center is not None and H is not None:
        board_x, board_y = image_to_board_coords(red_center, H)
        print(f"🎯 红球在棋盘坐标系位置: ({board_x:.1f}, {board_y:.1f})")
        cv2.putText(overlay, f"Board Pos: ({board_x:.0f}, {board_y:.0f})",
                    (red_center[1]+10, red_center[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # +++ Detect current ArUco markers and draw them

# 使用 ROI 加速 ArUco 检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)


    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        sorted_corners = [None] * 4
        for i, marker_id in enumerate(ids.flatten()):
            if 0 <= marker_id <= 3:
                pts = corners[i][0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                sorted_corners[marker_id] = (cx, cy)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {marker_id}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if None not in sorted_corners:
            current_corners = np.array(sorted_corners, dtype=np.float32)
            H, _ = cv2.findHomography(current_corners, ref_pts)
            H_inv = np.linalg.inv(H)
        else:
            H_inv = np.eye(3)
    else:
        H_inv = np.eye(3)


    
    # +++ Warp path using homography
    warped_path = []
    if refined_path:
        for pt in refined_path:
            # Convert (y,x) to (x,y) for homography
            px, py = pt[1], pt[0]
            warped_pt = cv2.perspectiveTransform(
                np.array([[[px, py]]], dtype=np.float32), H_inv).squeeze()
            warped_path.append((int(warped_pt[1]), int(warped_pt[0])))  # Back to (y,x)

    red_center = detect_red_ball(frame)
    display = cv2.addWeighted(path_overlay, 0.6, frame, 0.4, 0)

    if red_center:
        current_path = warped_path if warped_path else refined_path  # Fallback
        
        if current_path:
            distances = [calculate_distance(red_center, pt) for pt in current_path]
            nearest_idx = int(np.argmin(distances))
            target_idx = nearest_idx + headidx
            
            if target_idx < len(current_path):
                target_point = current_path[target_idx]

                # Visualization (original code)
                cv2.circle(display, (target_point[1], target_point[0]), 8, (255, 0, 255), -1)
                cv2.arrowedLine(display, (red_center[1], red_center[0]),
                               (target_point[1], target_point[0]), (255, 0, 255), 2)

                # Error tracking (original code)
                nearest_path_point = current_path[nearest_idx]
                euclidean_error = calculate_distance(red_center, nearest_path_point)
                errors.append(euclidean_error)
                timestamps.append(time.time())
                history_points.append(red_center)

                send_two_points_16bit(red_center[1], red_center[0],
                                      target_point[1], target_point[0])

    # ✅ 可视化 ROI 检测框（debug 用）
    if aruco_roi_bounds is not None:
        corners, ids = detect_aruco_in_roi(frame, aruco_roi_bounds)
    else:
        corners, ids, _ = aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMS)


    cv2.imshow("Red Ball Tracking", display)
    
    # Original key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        ret, new_frame = video_capture.read()
        if ret:
            path_overlay, red_center, refined_path = generate_path_overlay(new_frame)
            print("[P] Path regenerated")

video_capture.release()
cv2.destroyAllWindows()
