import matplotlib.pylab as plt  # I use version 3.1.2
# Notes for installing skimage: https://scikit-image.org/docs/dev/install.html
from skimage.morphology import skeletonize  # I use version 0.16.2
import numpy as np  # I use version 1.18.1
import cv2
import pandas as pd  # For reading the CSV file

# Load path coordinates from CSV
def load_path_coordinates(csv_file):
    df = pd.read_csv(csv_file, header=None)  # Assuming no header in CSV
    return df.values  # Returns a numpy array of shape (N, 2)

# Overlay path on the frame
def overlay_path(frame, path_coordinates):
    for i in range(len(path_coordinates) - 1):
        start_point = tuple(path_coordinates[i].astype(int))
        end_point = tuple(path_coordinates[i + 1].astype(int))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Green color, thickness 2
    return frame

# Your existing maze solving code
img_name = 'image4.jpg'
rgb_img = plt.imread(img_name)

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
path_coordinates = np.column_stack((path_x, path_y))  # Shape: (N, 2)

# Save to file (optional)
np.savetxt('maze_path_coordinates.csv', path_coordinates, delimiter=',')

# Load path coordinates from maze solver
path_coordinates = np.column_stack((path_x, path_y))  # From your first code

# Your existing ball detection code
lower_red_1 = np.array([0, 150, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 150, 100])
upper_red_2 = np.array([180, 255, 255])

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# --- Tuning parameters ---
scale_x = 0.7   # Change to 1.1 to stretch horizontally, <1 to shrink
scale_y = 0.7   # Change to 1.1 to stretch vertically
shift_x = 240     # Pixels to shift right (+) or left (-)
shift_y = 50     # Pixels to shift down (+) or up (-)
# Apply scale and shift
transformed_path = path_coordinates.copy()
transformed_path[:, 0] = transformed_path[:, 0] * scale_x + shift_x
transformed_path[:, 1] = transformed_path[:, 1] * scale_y + shift_y

while True:
    ret, frame = video_capture.read()
    if not ret: break
    # Get the width of the frame
    height, width = frame.shape[:2]

    # Crop the left 2/3 of the image
    crop_width = int(width)  # 2/3 of the width
    crop_height = int(height)
    cropped_frame = frame[:crop_height, :crop_width]  # Crop the image
   
    # Overlay the path on the cropped frame
    # Assuming path_coordinates is already scaled to match the cropped frame
# In overlay function
    cropped_frame = overlay_path(cropped_frame, transformed_path)


    # Your existing ball detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # ---------------------------------------------------------
            # NEW: Path comparison logic
            # 1. Find nearest point on path
            ball_pos = np.array([center_x, center_y])
            distances = np.linalg.norm(path_coordinates - ball_pos, axis=1)
            nearest_idx = np.argmin(distances)

            # 2. Get next target point (1 step ahead in path array)
            target_idx = min(nearest_idx + 1, len(path_coordinates) - 1)
            target_point = path_coordinates[target_idx]

            # 3. Calculate position difference
            error_x = target_point[0] - center_x
            error_y = target_point[1] - center_y
            # ---------------------------------------------------------

            # Draw visualization
            cv2.circle(frame, tuple(target_point.astype(int)), 7, (0, 0, 255), -1)  # Target
            cv2.arrowedLine(frame, (center_x, center_y),
                           (center_x + error_x, center_y + error_y),
                           (255, 0, 0), 2)  # Error vector

            # Your existing drawing
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"Error: ({error_x}, {error_y})", (center_x + 10, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print raw errors for motor control
            print(f"X_diff: {error_x}, Y_diff: {error_y}")

    cv2.imshow("Tracking (Cropped)", cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
