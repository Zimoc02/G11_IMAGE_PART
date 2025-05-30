import cv2
import numpy as np
import pandas as pd

# Load the CSV file containing the coordinates
csv_file = 'maze_path_coordinates.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Debug: Print column names to verify
print("Columns in CSV:", df.columns)

# Extract coordinates (adjust column names as needed)
coordinates = df[['X', 'Y']].values  # Replace 'X' and 'Y' with actual column names

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Main loop to display the camera feed with the path
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Draw the path on the frame
    for i in range(1, len(coordinates)):
        cv2.line(frame, tuple(coordinates[i-1]), tuple(coordinates[i]), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Camera Feed with Path', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
