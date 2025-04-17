import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()
frame = picam2.capture_array()
cv2.imwrite("photo.jpg", cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
picam2.stop()
