import cv2
import numpy as np

# Load MiDaS depth estimation model
model = cv2.dnn.readNet("model-small.onnx")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Initialize webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Thresholds (adjust these interactively)
lower_threshold = 100  # Holes (darker regions)
upper_threshold = 200  # Walls (brighter regions)

cv2.namedWindow("Segmentation")
cv2.createTrackbar("Hole Threshold", "Segmentation", lower_threshold, 255, lambda x: None)
cv2.createTrackbar("Wall Threshold", "Segmentation", upper_threshold, 255, lambda x: None)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    model.setInput(blob)
    depth_map = model.forward()

    # Fix: Remove extra dimensions
    depth_map = np.squeeze(depth_map)  # Critical fix here!

    # Post-process
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Get current threshold values
    lower_threshold = cv2.getTrackbarPos("Hole Threshold", "Segmentation")
    upper_threshold = cv2.getTrackbarPos("Wall Threshold", "Segmentation")

    # Segment layers
    _, holes_mask = cv2.threshold(depth_map, lower_threshold, 255, cv2.THRESH_BINARY_INV)
    _, walls_mask = cv2.threshold(depth_map, upper_threshold, 255, cv2.THRESH_BINARY)
    middle_mask = cv2.bitwise_and(depth_map, depth_map, mask=~cv2.bitwise_or(holes_mask, walls_mask))

    # Visualization
    output = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
    output[holes_mask > 0] = (0, 0, 255)  # Red = Holes
    output[walls_mask > 0] = (255, 0, 0)  # Blue = Walls
    output[middle_mask > 0] = (0, 255, 0) # Green = Middle (path)

    cv2.imshow("Segmentation", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
