import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import math
import smbus
from collections import deque
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import time
import csv
import os
import sys
import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cv2 import aruco
from tkinter import messagebox

# ========== 原始参数和设置保持不变 ==========
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters_create()
bus = smbus.SMBus(1)
arduino_address = 0x08

INTERPOLATION_COUNT = 2
headidx = 30
lower_red_1 = np.array([0, 120, 20])
upper_red_1 = np.array([10, 255, 120])
lower_red_2 = np.array([170, 120, 20])
upper_red_2 = np.array([180, 255, 120])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
last_send_time = 0
SEND_INTERVAL = 0.04

errors = []
timestamps = []
history_points = deque(maxlen=300)
nearest_points = []
aruco_corners_dict = {}
aruco_roi_list = []
aruco_centers_dict = {}

real_world_path = []
path_overlay = None
H = None

video_running = False
video_thread = None
video_capture = None
canvas = None
photo = None
panel = None

red_center = None
target_point = None
real_world_red = None

start_time = None  # 程序运行起始时间（新增）


def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val
    return [(val >> 8) & 0xFF, val & 0xFF]


def send_two_points_16bit(x1, y1, x2, y2):
    global last_send_time
    now = time.time()
    if now - last_send_time < SEND_INTERVAL:
        return
    last_send_time = now
    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
    except:
        pass


# ========== 原始图像处理逻辑中的函数保持不变 ==========
# ...
# 为节省篇幅，略去 `generate_path_overlay`, `detect_red_ball`, `compute_homography_from_aruco`, 等函数
# 请将这些函数原封不动地复制粘贴进来
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
            used_indices.update(kdtree.query_ball_point(point, r=10))
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
            if candidate not in used and calculate_distance(current_point, candidate) < 120:
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



# ========== GUI 控制逻辑 ===========

def start_video():
    global video_running, video_thread, start_time, video_capture
    video_running = True
    start_time = time.time()
    video_capture = cv2.VideoCapture(0)

    # 路径识别一次
    ret, frame = video_capture.read()
    if ret:
        frame = cv2.flip(frame, 1)  # 同样加翻转
        overlay, red_center_, refined_path = generate_path_overlay(frame)
        get_aruco_centers(aruco_corners_dict)
        H = compute_homography_from_aruco(aruco_centers_dict, target_size=(22, 17))
        if refined_path and H is not None:
            real_world_path.clear()
            real_world_path.extend(map_path_to_aruco_plane_coords(refined_path, H))
        else:
            messagebox.showwarning("⚠️", "初始路径识别失败，请检查图像与ArUco标签")
    else:
        messagebox.showwarning("⚠️", "无法从摄像头读取初始图像")

    video_thread = threading.Thread(target=video_loop)
    video_thread.start()

def update_runtime():
    if video_running and start_time:
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        runtime_value.config(text=f"{h:02}:{m:02}:{s:02}")
    root.after(1000, update_runtime)

path_display_visible = False
path_canvas = None
path_fig = None
path_ax = None

def toggle_path_display():
    global path_display_visible, path_canvas, path_fig, path_ax
    if not real_world_path:
        messagebox.showwarning("提示", "路径数据为空！")
        return

    if not path_display_visible:
        path_fig = plt.Figure(figsize=(4, 4), dpi=100)
        path_ax = path_fig.add_subplot(111)
        path_canvas = FigureCanvasTkAgg(path_fig, master=info_frame)
        path_canvas.get_tk_widget().pack(pady=10)
        path_display_visible = True
        update_path_dot_only()
    else:
        path_canvas.get_tk_widget().pack_forget()
        path_display_visible = False

def update_path_dot_only():
    if path_display_visible and path_ax:
        path_ax.clear()
        if real_world_path:
            x_vals, y_vals = zip(*real_world_path)
            path_ax.plot(x_vals, y_vals, label='Path')
        if real_world_red:
            path_ax.scatter(real_world_red[0], real_world_red[1], c='r', s=60, label='Ball')
        path_ax.set_title("Ball Path")
        path_ax.set_xlabel("X (cm)")
        path_ax.set_ylabel("Y (cm)")
        path_ax.legend()
        path_ax.grid(True)
        path_ax.axis('equal')
        path_canvas.draw()
    root.after(500, update_path_dot_only)

def plot_error_graph():
    try:
        duration = int(duration_entry.get())
        if not timestamps:
            messagebox.showwarning("提示", "没有误差数据！")
            return

        current_time = time.time() - start_time
        target_start_time = current_time - duration

        x_vals = []
        y_vals = []

        for t, e in zip(timestamps, errors):
            if t >= target_start_time:
                x_vals.append(t - target_start_time)
                y_vals.append(e)

        if not x_vals:
            messagebox.showwarning("提示", "指定时间段内无数据")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, marker='o')
        plt.title(f"Tracking Error (Last {duration} Seconds)")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (cm)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except ValueError:
        messagebox.showerror("输入错误", "请输入一个有效的秒数！")

def stop_video():
    global video_running
    video_running = False

def video_loop():
    global video_capture, red_center, target_point, real_world_red

    video_capture = cv2.VideoCapture(0)
    while video_running:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️ 无法读取视频帧")
            break

        frame = cv2.flip(frame, 1)

        # 每帧只找红球，不重新找路径
        red_center = detect_red_ball(frame)
        if red_center is not None:
            real_world_red = detect_aruco_and_map_red_ball(red_center, frame)

            if real_world_red:
                current_value.config(text=f"({real_world_red[0]:.2f}, {real_world_red[1]:.2f})")

                if real_world_path:
                    dists = [calculate_distance(real_world_red, pt) for pt in real_world_path]
                    min_idx = np.argmin(dists)
                    error_point = real_world_path[min_idx]   # 最近点
                    error = dists[min_idx]

                    # 找领先30个点作为目标点
                    target_idx = min(min_idx + headidx, len(real_world_path) - 1)
                    target_point = real_world_path[target_idx]

                    # 显示与保存
                    target_value.config(text=f"({target_point[0]:.2f}, {target_point[1]:.2f})")
                    errors.append(error)
                    timestamps.append(time.time() - start_time)

                    # I2C发送
                    x1, y1 = int(real_world_red[0] * 10), int(real_world_red[1] * 10)
                    x2, y2 = int(target_point[0] * 10), int(target_point[1] * 10)
                    #send_two_points_16bit(x1, y1, x2, y2)

        # 显示视频帧
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.config(image=imgtk)

    video_capture.release()
    print("🎥 视频线程退出")

def regenerate_path():
    global real_world_path, path_overlay, H
    if video_capture is not None:
        ret, frame = video_capture.read()
        if ret:
            path_overlay, red_center_, refined_path = generate_path_overlay(frame)
            get_aruco_centers(aruco_corners_dict)
            H = compute_homography_from_aruco(aruco_centers_dict, target_size=(22, 17))
            real_world_path = map_path_to_aruco_plane_coords(refined_path, H)
        else:
            messagebox.showwarning("提示", "无法从摄像头读取图像帧！")
    else:
        messagebox.showinfo("提示", "视频未启动，请先点击 Start")

def save_accuracy():
    if not errors or not timestamps:
        messagebox.showinfo("提示", "没有可保存的误差数据")
        return
    try:
        filename = time.strftime("accuracy_%Y%m%d_%H%M%S.csv")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp (s)', 'Error (cm)'])
            for t, e in zip(timestamps, errors):
                writer.writerow([t, e])
        messagebox.showinfo("保存成功", f"准确度数据已保存为 {filename}")
    except Exception as e:
        messagebox.showerror("保存失败", str(e))


# ========== 退出钩子 ==========
def on_closing():
    if messagebox.askokcancel("退出确认", "确定要退出程序吗？"):
        global video_running
        video_running = False
        if video_capture:
            video_capture.release()
        root.destroy()

# ========== 启动 GUI ==========
root = tk.Tk()
root.title("红球追踪 GUI 合并版")
root.geometry("1280x720")

# 左侧按钮栏
control_frame = tk.Frame(root)
control_frame.pack(side="left", fill="y", padx=10, pady=10)

tk.Button(control_frame, text="▶️ Start", width=20, height=2, command=start_video).pack(pady=5)
tk.Button(control_frame, text="⏹️ Stop", width=20, height=2, command=stop_video).pack(pady=5)
tk.Button(control_frame, text="🔁 Re-generate Path", width=20, height=2, command=regenerate_path).pack(pady=5)
tk.Button(control_frame, text="💾 Save Accuracy", width=20, height=2, command=save_accuracy).pack(pady=5)

# 实时运行时间显示
runtime_label = tk.Label(control_frame, text="运行时间:", font=("Arial", 12))
runtime_label.pack(pady=(30, 0))
runtime_value = tk.Label(control_frame, text="00:00:00", font=("Arial", 12), fg="purple")
runtime_value.pack()

# 实时误差图时间段输入框
tk.Label(control_frame, text="误差图时长 (秒):", font=("Arial", 12)).pack(pady=(20, 0))
duration_entry = tk.Entry(control_frame, width=10)
duration_entry.pack()
tk.Button(control_frame, text="📈 生成误差图", width=20, command=plot_error_graph).pack(pady=5)

# 加载路径 + 当前点嵌入显示开关
tk.Button(control_frame, text="📍 查看路径&小球", width=20, command=toggle_path_display).pack(pady=20)

# 右侧主区域，包含视频和信息面板
right_frame = tk.Frame(root)
right_frame.pack(side="right", fill="both", expand=True)

# 视频显示区域
video_frame = tk.Frame(right_frame)
video_frame.pack(side="left", fill="both", expand=True)

panel = tk.Label(video_frame)
panel.pack(fill="both", expand=True)

# 坐标显示区域
info_frame = tk.Frame(right_frame, bd=2, relief="groove")
info_frame.pack(side="right", padx=10, pady=10, fill="y")

tk.Label(info_frame, text="🎯 当前坐标 (X, Y):", font=("Arial", 12)).grid(row=0, column=0, sticky="w")
current_value = tk.Label(info_frame, text="(0.00, 0.00)", font=("Arial", 12), fg="blue")
current_value.grid(row=0, column=1, sticky="w")

tk.Label(info_frame, text="🏁 目标坐标 (X, Y):", font=("Arial", 12)).grid(row=1, column=0, sticky="w")
target_value = tk.Label(info_frame, text="(0.00, 0.00)", font=("Arial", 12), fg="green")
target_value.grid(row=1, column=1, sticky="w")

root.protocol("WM_DELETE_WINDOW", on_closing)
update_runtime()
root.mainloop()
