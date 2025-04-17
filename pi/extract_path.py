import cv2
import numpy as np
import json

def extract_maze_path(image_path, output_json="path_data.json"):
    # Load maze image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to isolate black path
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Remove numbers/text using morphological opening
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (the path)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("Error: No contours found")
        return
    
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.squeeze()
    
    # Create dense path points
    dense_path = []
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        num_points = int(np.linalg.norm(p2 - p1))
        if num_points > 0:
            new_points = np.linspace(p1, p2, num_points).astype(int)
            dense_path.extend(new_points.tolist())  # Convert numpy array to list
    
    # Convert to NumPy array and save
    dense_array = np.array(dense_path)
    with open(output_json, 'w') as f:
        json.dump(dense_array.tolist(), f)  # Now safe to use .tolist()
    
    print(f"Path extracted with {len(dense_array)} points. Saved to {output_json}")

if __name__ == "__main__":
    extract_maze_path("maze_path.jpeg")  # Replace with your image path
