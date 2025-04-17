import matplotlib.pylab as plt
from skimage.morphology import skeletonize
import numpy as np
import cv2
import pandas as pd
import smbus

# I2C Configuration
bus = smbus.SMBus(1)
arduino_address = 0x08

def int16_to_bytes(val):
    val = int(val)
    if val < 0:
        val = (1 << 16) + val
    return [(val >> 8) & 0xFF, val & 0xFF]

def send_two_points_16bit(x1, y1, x2, y2):
    data = int16_to_bytes(x1) + int16_to_bytes(y1) + int16_to_bytes(x2) + int16_to_bytes(y2)
    try:
        bus.write_i2c_block_data(arduino_address, 0x00, data)
        print(f"Sent: ({x1}, {y1}), ({x2}, {y2})")
    except Exception as e:
        print(f"I2C Send Error: {e}")

# Path processing functions
def load_path_coordinates(csv_file):
    df = pd.read_csv(csv_file, header=None)
    return df.values

def overlay_path(frame, path_coordinates):
    for i in range(len(path_coordinates) - 1):
        start = tuple(path_coordinates[i].astype(int))
        end = tuple(path_coordinates[i+1].astype(int))
        cv2.line(frame, start, end, (0, 255, 0), 2)
    return frame

# Maze solving code (run once during initialization)
img_name = 'image4.jpg'
rgb_img = plt.imread(img_name)

# [Keep the original maze solving code here...]
# ... (The maze solving code from your original implementation) ...
plt.figure(figsize=(14, 14))
plt.imshow(rgb_img)
x0, y0 = 700, 80  # Start x point
x1, y1 = 1050, 500  # Start y point

plt.plot(x0, y0, 'gx', markersize=14)
plt.plot(x1, y1, 'rx', markersize=14)
if rgb_img.shape.__len__() > 2:
    thr_img = rgb_img[:, :, 0] > np.max(rgb_img[:, :, 0]) / 2
else:
    thr_img = rgb_img > np.max(rgb_img) / 2
skeleton = skeletonize(thr_img)
plt.figure(figsize=(14, 14))
plt.imshow(skeleton)
# Map of routes
mapT = ~skeleton
plt.imshow(mapT)
plt.plot(x0, x0, 'gx', markersize=14)
plt.plot(x1, y1, 'rx', markersize=14)
_mapt = np.copy(mapT)

# Searching for our end point and connect to the path
boxr = 30

# Just a little safety check, if the points are too near the edge, it will error
if y1 < boxr: y1 = boxr
if x1 < boxr: x1 = boxr

cpys, cpxs = np.where(_mapt[y1 - boxr:y1 + boxr, x1 - boxr:x1 + boxr] == 0)
# Calibrate points to main scale
cpys += y1 - boxr
cpxs += x1 - boxr
# Find closest point of possible path end points
idx = np.argmin(np.sqrt((cpys - y1) ** 2 + (cpxs - x1) ** 2))
y, x = cpys[idx], cpxs[idx]

pts_x = [x]
pts_y = [y]
pts_c = [0]

# Mesh of displacements
xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
ymesh = ymesh.reshape(-1)
xmesh = xmesh.reshape(-1)

dst = np.zeros((thr_img.shape))

# Breath first algorithm exploring a tree
while True:
    # Update distance
    idc = np.argmin(pts_c)
    ct = pts_c.pop(idc)
    x = pts_x.pop(idc)
    y = pts_y.pop(idc)
    # Search 3x3 neighbourhood for possible
    ys, xs = np.where(_mapt[y - 1:y + 2, x - 1:x + 2] == 0)
    # Invalidate these points from future searches
    _mapt[ys + y - 1, xs + x - 1] = ct
    _mapt[y, x] = 9999999
    # Set the distance in the distance image
    dst[ys + y - 1, xs + x - 1] = ct + 1
    # Extend our list
    pts_x.extend(xs + x - 1)
    pts_y.extend(ys + y - 1)
    pts_c.extend([ct + 1] * xs.shape[0])
    # If we run out of points
    if pts_x == []:
        break
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < boxr:
        edx = x
        edy = y
        break
plt.figure(figsize=(14, 14))
plt.imshow(dst)

path_x = []
path_y = []

y = edy
x = edx
# Traces best path
while True:
    nbh = dst[y - 1:y + 2, x - 1:x + 2]
    nbh[1, 1] = 9999999
    nbh[nbh == 0] = 9999999
    # If we reach a dead end
    if np.min(nbh) == 9999999:
        break
    idx = np.argmin(nbh)
    # Find direction
    y += ymesh[idx]
    x += xmesh[idx]

    if np.sqrt((x - x1) ** 2 + (y - y1) ** 2) < boxr:
        print('Optimum route found.')
        break
    path_y.append(y)
    path_x.append(x)
plt.figure(figsize=(14, 14))
plt.imshow(rgb_img)
plt.plot(path_x, path_y, 'r-', linewidth=5)
# After maze solving:
path_coordinates = np.column_stack((path_x, path_y))
np.savetxt('maze_path_coordinates.csv', path_coordinates, delimiter=',')

# Video processing parameters
LOWER_RED_1 = np.array([0, 150, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 150, 100])
UPPER_RED_2 = np.array([180, 255, 255])
MIN_CONTOUR_AREA = 500
ROI_SIZE = 50  # Search area radius around last known position

# Tracking state
last_ball_pos = None
tracking_lost_count = 0

# Main video processing loop
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Frame preprocessing
    height, width = frame.shape[:2]
    crop_width = int(width)
    crop_height = int(height)
    cropped_frame = frame[:crop_height, :crop_width]
    display_frame = cropped_frame.copy()
    display_frame = overlay_path(display_frame, path_coordinates)

    # Region of Interest management
    roi = None
    if last_ball_pos is not None and tracking_lost_count < 5:
        x, y = last_ball_pos
        x_min = max(0, x - ROI_SIZE)
        x_max = min(crop_width, x + ROI_SIZE)
        y_min = max(0, y - ROI_SIZE)
        y_max = min(crop_height, y + ROI_SIZE)
        roi = (x_min, y_min, x_max - x_min, y_max - y_min)
        roi_frame = cropped_frame[y_min:y_max, x_min:x_max]
    else:
        roi_frame = cropped_frame

    # Ball detection in ROI
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1),
                         cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2))
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_found = False

    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Adjust coordinates if using ROI
            if roi is not None:
                x += roi[0]
                y += roi[1]
            
            center_x = x + w//2
            center_y = y + h//2
            last_ball_pos = (center_x, center_y)
            tracking_lost_count = 0
            ball_found = True
            break

    if not ball_found:
        tracking_lost_count += 1
        last_ball_pos = None

    # Path following logic
    if last_ball_pos is not None:
        center_x, center_y = last_ball_pos
        
        # Find nearest path point
        distances = np.linalg.norm(path_coordinates - (center_x, center_y), axis=1)
        nearest_idx = np.argmin(distances)
        
        # Look 5 points ahead on the path
        target_idx = min(nearest_idx + 10, len(path_coordinates)-1)
        target_point = path_coordinates[target_idx]
        error_x = target_point[0] - center_x
        error_y = target_point[1] - center_y

        # Send data to Arduino
        send_two_points_16bit(center_x, center_y, target_point[0], target_point[1])

        # Visual feedback
        cv2.circle(display_frame, (center_x, center_y), 7, (255, 0, 0), -1)
        cv2.circle(display_frame, tuple(target_point.astype(int)), 7, (0, 0, 255), -1)
        cv2.arrowedLine(display_frame, (center_x, center_y), 
                       (center_x + error_x, center_y + error_y),
                       (255, 0, 255), 2)
        cv2.putText(display_frame, f"Error: ({error_x}, {error_y})", 
                   (center_x + 10, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show ROI area
    if roi is not None:
        cv2.rectangle(display_frame, 
                     (roi[0], roi[1]),
                     (roi[0] + roi[2], roi[1] + roi[3]),
                     (0, 255, 255), 2)

    cv2.imshow("Ball Tracking", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
