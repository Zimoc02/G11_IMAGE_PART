import cv2

# Initialize the camera with optimized settings
video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 15)

# Check if the camera opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to play video
def play_video():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Start playing video
play_video()
