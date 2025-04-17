import cv2
import numpy as np
import math
import smbus
from collections import deque
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import time  # 加在最顶部和其他 import 放一起
import csv

from cv2 import aruco
#Aruco 设置
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters_create()

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
nearest_points = []  # ✅ 新增记录最近路径点
aruco_corners_dict = {}  # 用于存储四个ArUco marker的角点
aruco_roi_list = [] #ArUco ROI初始化
aruco_centers_dict = {}  # 用于保存每个 ArUco ID 的中心点坐标

last_red_center = None  # 👈 放在全局变量区域

x__1 = 0
x__2 = 0
y__1 = 0
y__2 = 0

# 实时识别红球位置
def detect_red_ball(frame):
    global last_red_center
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 默认使用整张图像
    h, w = frame.shape[:2]
    roi_margin = 60

    if last_red_center:
        yc, xc = last_red_center
        y_min = max(0, yc - roi_margin)
        y_max = min(h, yc + roi_margin)
        x_min = max(0, xc - roi_margin)
        x_max = min(w, xc + roi_margin)

        roi_hsv = hsv[y_min:y_max, x_min:x_max]
        red_mask = cv2.bitwise_or(
            cv2.inRange(roi_hsv, lower_red_1, upper_red_1),
            cv2.inRange(roi_hsv, lower_red_2, upper_red_2)
        )
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_center = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_area:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                best_center = (y + h_box // 2 + y_min, x + w_box // 2 + x_min)  # 加偏移
                max_area = area

        if best_center is not None:
            last_red_center = best_center
            return best_center

    # Fallback：整图查找
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red_1, upper_red_1),
        cv2.inRange(hsv, lower_red_2, upper_red_2)
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_center = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100 and area > max_area:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            best_center = (y + h_box // 2, x + w_box // 2)
            max_area = area

    if best_center is not None:
        last_red_center = best_center
    return best_center

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


# 路径图层函数
def generate_path_overlay(image):
    '''
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

        if 50 < area < 1500 and perimeter > 80:
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) > 5:
                cv2.drawContours(selected_path_image, [cnt], -1, (255, 255, 255), 2)
    
  
    # 骨架提取
    gray_selected = cv2.cvtColor(selected_path_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_selected, 127, 255, cv2.THRESH_BINARY)
    skeleton = cv2.ximgproc.thinning(binary)
    non_zero_points = np.column_stack(np.where(skeleton > 0))
    '''
    global aruco_corners_dict

    # 基于颜色的黑线掩膜提取路径（避免识别到洞）    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 骨架提取（黑线细化）
    skeleton = cv2.ximgproc.thinning(black_mask)
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
            #if candidate not in used and calculate_distance(current_point, candidate) < 65:
            if candidate not in used and calculate_distance(current_point, candidate) < 100:
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

    # === [ArUco 检测逻辑，并且设置四个ROI]  ===
    aruco_roi_list.clear()#清理可能残留的ROI信息
    margin = 80 #设置控制用于检测ArUco的ROI大小的变量
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is None:
        print("路线检测时没有检测到 任何ArUco")
        return []
    else:
        for i, marker_id in enumerate(ids.flatten()):
            if 0 <= marker_id <= 3:
                pts = corners[i][0]  # shape: (4, 2)
                aruco_corners_dict[marker_id] = pts
                x_min = max(0, int(np.min(pts[:, 0])) - margin)
                x_max = min(image.shape[1], int(np.max(pts[:, 0])) + margin)
                y_min = max(0, int(np.min(pts[:, 1])) - margin)
                y_max = min(image.shape[0], int(np.max(pts[:, 1])) + margin)
                aruco_roi_list.append((marker_id, (x_min, x_max, y_min, y_max)))

    if len(aruco_roi_list) < 4:
        print("⚠️ 检测到的 ArUco 不完整，可能不包含全部 0~3")
    else:
        print("✅ 初始 ArUco ROI 区域提取完成") 

    if ids is not None:
        aruco.drawDetectedMarkers(overlay, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(overlay, f"ID: {marker_id}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return overlay, red_center, refined_path
#将四个角坐标转换为中心坐标
def get_aruco_centers(aruco_corners_dict):
    global aruco_centers_dict
    aruco_centers_dict.clear()
    for marker_id, corners in aruco_corners_dict.items():
        center = np.mean(corners, axis=0)
        aruco_centers_dict[marker_id] = tuple(center)

#生成 H 矩阵 加入中心点转化
def compute_homography_from_aruco(aruco_centers_dict, target_size=(22, 17)):
    if len(aruco_centers_dict) < 4:
        print("⚠️ ArUco 中心点不足，无法生成 H 矩阵")
        return None

    # 1. 提取四个 marker 中心点（按照你指定的顺序）
    try:
        src_pts = np.array([
            aruco_centers_dict[0],
            aruco_centers_dict[1],
            aruco_centers_dict[2],
            aruco_centers_dict[3]
        ], dtype=np.float32)
    except KeyError as e:
        print(f"⚠️ 缺失 ArUco {e}，请确认0~3全部识别到了")
        return None

    # 2. 定义目标坐标点（你希望他们对应的位置，比如一个 400x400 的棋盘）
    dst_pts = np.array([
        [-(target_size[0]/2), (target_size[1]/2)],             # ArUco 0 → 左上角
        [(target_size[0]/2), (target_size[1]/2)],       # ArUco 1 → 右上角
        [(target_size[0]/2), -(target_size[1]/2)], # ArUco 2 → 右下角
        [-(target_size[0]/2), -(target_size[1]/2)]        # ArUco 3 → 左下角
    ], dtype=np.float32)

    # 3. 计算 Homography
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H
#实时检测ARUCO的位置（四个 ROI 中的 ）
def detect_aruco_and_update(frame):
    global aruco_corners_dict, aruco_centers_dict, aruco_roi_list
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 如果 ROI 区域存在，先进行 ROI 裁剪并只在该区域检测
    if aruco_roi_list:
        for roi in aruco_roi_list:
            marker_id, (x_min, x_max, y_min, y_max) = roi
            roi_frame = gray[y_min:y_max, x_min:x_max]
            corners, ids, _ = aruco.detectMarkers(roi_frame, ARUCO_DICT, parameters=ARUCO_PARAMS)

            # 检测到 ArUco marker 后更新字典
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if 0 <= marker_id <= 3:
                        pts = corners[i][0] + np.array([x_min, y_min])  # 添加 ROI 偏移
                        aruco_corners_dict[marker_id] = pts
                        center = np.mean(pts, axis=0)
                        aruco_centers_dict[marker_id] = tuple(center)

        print(f"🔍 实时检测到 ArUco ID: {list(aruco_corners_dict.keys())}")
    else:
        # 如果没有 ROI，直接在整个图像上进行检测
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is not None:
            aruco_corners_dict.clear()
            aruco_centers_dict.clear()
            for i, marker_id in enumerate(ids.flatten()):
                if 0 <= marker_id <= 3:
                    pts = corners[i][0]
                    aruco_corners_dict[marker_id] = pts
                    center = np.mean(pts, axis=0)
                    aruco_centers_dict[marker_id] = tuple(center)
            print(f"🔍 实时检测到 ArUco ID: {list(aruco_corners_dict.keys())}")
        else:
            print("⚠️ 实时未检测到任何 ArUco")

#使用识别出来的ArUco进行小球坐标还原            
def detect_aruco_and_map_red_ball(red_center, frame):
    """
    已知红球图像坐标 red_center，
    自动检测 ArUco marker、计算 H 矩阵，并将 red_center 映射到 ArUco 平面坐标系中。
    
    参数：
        red_center: tuple(int, int)，图像坐标下的红球位置 (y, x)
        frame: 当前图像帧
    
    返回：
        real_world_red_center: tuple(float, float)，映射到 ArUco 平面坐标系的坐标 (x, y)
    """
    if red_center is None:
        print("⚠️ 输入的 red_center 为空，跳过")
        return None

    # 1. 实时检测 ArUco marker 并更新 ROI 和角点字典
    detect_aruco_and_update(frame)

    # 2. 获取中心点
    get_aruco_centers(aruco_corners_dict)

    # 3. 计算 Homography 矩阵
    H = compute_homography_from_aruco(aruco_centers_dict, target_size=(22, 17))
    if H is None:
        print("❌ 无法计算 Homography，返回 None")
        return None

    # 4. 将红球图像坐标转换为 ArUco 平面坐标系
    point_arr = np.array([[[red_center[1], red_center[0]]]], dtype=np.float32)  # 注意顺序 x, y
    projected = cv2.perspectiveTransform(point_arr, H)
    real_world_red_center = tuple(projected[0][0])

    return real_world_red_center


def map_path_to_aruco_plane_coords(refined_path, H):
    if H is None:
        print("❌ H 矩阵为空，无法进行坐标映射")
        return []

    try:
        # 构造路径点数组，注意图像坐标是 (y, x)，需要变为 (x, y)
        path_array = np.array([[[p[1], p[0]]] for p in refined_path], dtype=np.float32)

        # 透视变换
        projected = cv2.perspectiveTransform(path_array, H)

        # 提取结果
        mapped_path = [tuple(pt[0]) for pt in projected]  # [(x1, y1), (x2, y2), ...]
        return mapped_path

    except Exception as e:
        print(f"❌ 路径映射失败: {e}")
        return []
def inverse_homography_point(pt, H):
    """
    将物理坐标系中的点 (x, y) 通过 H⁻¹ 映射回图像坐标系 (u, v)
    """
    inv_H = np.linalg.inv(H)
    pt_arr = np.array([[[pt[0], pt[1]]]], dtype=np.float32)  # 注意是 [[(x, y)]]
    img_pt = cv2.perspectiveTransform(pt_arr, inv_H)
    return int(img_pt[0][0][0]), int(img_pt[0][0][1])


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

#开始使用Homography转化路径
get_aruco_centers(aruco_corners_dict)
H = compute_homography_from_aruco(aruco_centers_dict, target_size=(22, 17))
if H is not None:
    print("✅ Homography 矩阵计算完成")
    print(H)
real_world_path = map_path_to_aruco_plane_coords(refined_path, H)
# 导出路径坐标点（单位是以 ArUco 平面中心为 (0, 0) 的实际坐标系）
with open('real_world_path.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X (cm)', 'Y (cm)'])  # 或者单位是你设置的格子单位
    for x, y in real_world_path:
        writer.writerow([x, y])
print("✅ 已保存路径到 real_world_path.csv")

'''
#得到 H 后，在 refined_path 上做投影
if H is not None:
    path_array = np.array(refined_path, dtype=np.float32)
    path_array = np.array([[[p[1], p[0]]] for p in refined_path], dtype=np.float32)  # 注意顺序：x, y
    projected_path = cv2.perspectiveTransform(path_array, H)
    real_world_path = [(float(p[0][0]), float(p[0][1])) for p in projected_path]  # 棋盘单位下的路径点
'''
real_world_red = None
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    red_center = detect_red_ball(frame)
    display = cv2.addWeighted(path_overlay, 0.6, frame, 0.4, 0)

    if red_center:
        real_world_red = detect_aruco_and_map_red_ball(red_center, frame)
    #if real_world_red:
     #   print(f"红球在 ArUco 平面坐标系中的位置: {real_world_red}")

    if real_world_red and real_world_path:
        # === 在物理坐标系中计算距离 ===
        distances = [calculate_distance(real_world_red, pt) for pt in real_world_path]
        nearest_idx = int(np.argmin(distances))
        target_idx = nearest_idx + headidx

        if target_idx < len(real_world_path):
            target_point = real_world_path[target_idx]
            print(f"[Current] 红球: X={real_world_red[0]:.2f}, Y={real_world_red[1]:.2f}")
            print(f"[Target ] 目标: X={target_point[0]:.2f}, Y={target_point[1]:.2f}")

            # === 映射目标点到图像坐标用于可视化 ===
            target_px = inverse_homography_point(target_point, H)

            # 在图像中画目标点
            cv2.circle(display, target_px, 8, (255, 0, 255), -1)
            cv2.putText(display, "Target", (target_px[0] + 10, target_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # === 误差统计（仍然使用物理坐标距离） ===
            nearest_path_point = real_world_path[nearest_idx]
            nearest_points.append(nearest_path_point)

            euclidean_error = calculate_distance(real_world_red, nearest_path_point)
            errors.append(euclidean_error)
            timestamps.append(time.time())
            history_points.append(real_world_red)

            if len(errors) % 30 == 0:
                print(f"[误差统计] 平均: {np.mean(errors):.2f} cm, 最大: {np.max(errors):.2f} cm, 标准差: {np.std(errors):.2f} cm")

            # === 控制输出坐标（如果需要发送） ===
            x__1 = real_world_red[0]
            y__1 = real_world_red[1]
            x__2 = target_point[0]
            y__2 = target_point[1]

                
    cv2.imshow("Red Ball Tracking", display)

    if errors:
        current_error = errors[-1]
        cv2.putText(display, f"Error: {int(current_error)} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        ret, new_frame = video_capture.read()
        if ret:
            path_overlay, red_center, refined_path = generate_path_overlay(new_frame)
            real_world_path = map_path_to_aruco_plane_coords(refined_path, H)
            print("[热键P] 重新生成路径图层")
    elif key == ord('s'):
        with open('tracking_accuracy.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Error (cm)', 
                         'Red_X (cm)', 'Red_Y (cm)', 
                         'Nearest_X (cm)', 'Nearest_Y (cm)'])
            for t, e, red_pt, path_pt in zip(timestamps, errors, history_points, nearest_points):
                writer.writerow([
                f"{t:.3f}",
                f"{e:.3f}",
                f"{red_pt[0]:.3f}", f"{red_pt[1]:.3f}",
                f"{path_pt[0]:.3f}", f"{path_pt[1]:.3f}"
                ])
        print("✅ [热键S] 已保存物理空间误差数据到 tracking_accuracy.csv")



video_capture.release()
cv2.destroyAllWindows()
