import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import math

# ========== Hardware Constants ==========
BOARD_WIDTH_CM = 23
BOARD_HEIGHT_CM = 16
HOLE_DIAMETER_CM = 1.1
CAM_RES = (1280, 720)

# ========== Calibration Constants ==========
PX_TO_CM_X = BOARD_WIDTH_CM / CAM_RES[0]  # cm per pixel
PX_TO_CM_Y = BOARD_HEIGHT_CM / CAM_RES[1]
HOLE_RADIUS_PX = (HOLE_DIAMETER_CM/2) / PX_TO_CM_X

# ========== Control Parameters ==========
D1 = 0.04  # Distance weight
D2 = 0.05  # Approach speed weight
STUCK_VELOCITY_THRESH = 0.1  # cm/s
INTEGRATION_RATE = np.deg2rad(5)  # 5 degrees per step

# ========== Tracking System Components ==========
class BallTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.integrated_angle = 0.0
        self.last_update = cv2.getTickCount()

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0,100,50), (10,255,255)) | \
               cv2.inRange(hsv, (170,100,50), (180,255,255))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_state = None
        vel_cm = (0.0, 0.0)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(largest)
            measurement = np.array([[x], [y]], dtype=np.float32)
            
            self.kalman.correct(measurement)
            predicted = self.kalman.predict()
            
            vel = self.kalman.statePost[2:4].flatten()
            vel_cm = (vel[0] * PX_TO_CM_X, vel[1] * PX_TO_CM_Y)
            ball_state = (predicted[0], predicted[1])

            if np.linalg.norm(vel_cm) < STUCK_VELOCITY_THRESH:
                delta_time = (cv2.getTickCount() - self.last_update)/cv2.getTickFrequency()
                if delta_time > 2.0:
                    self.integrated_angle += INTEGRATION_RATE
            else:
                self.integrated_angle = 0.0
                self.last_update = cv2.getTickCount()

        return ball_state, vel_cm

class HoleManager:
    def __init__(self):
        self.holes = [(535,173), (443,172), (364,180)]
        self.hole_color = (0, 0, 255)
        self.danger_color = (0, 255, 255)

    def check_collision(self, pos_px, vel_cm):
        risk_vector = np.zeros(2)
        pos_px = np.array(pos_px)
        
        for hole in self.holes:
            hole_arr = np.array(hole)
            displacement = pos_px - hole_arr
            dist = np.linalg.norm(displacement)
            
            if dist == 0:
                continue
                
            direction = displacement / dist
            vel_px = (vel_cm[0]/PX_TO_CM_X, vel_cm[1]/PX_TO_CM_Y)
            approach_speed = np.dot(vel_px, direction)
            
            if dist < 3*HOLE_RADIUS_PX and approach_speed < 0:
                alpha = D1*(1 - dist/(3*HOLE_RADIUS_PX)) - D2*abs(approach_speed)
                risk_vector += alpha * direction
                
        return risk_vector

    def draw_holes(self, frame):
        for hole in self.holes:
            cv2.circle(frame, hole, int(HOLE_RADIUS_PX), self.hole_color, -1)
            cv2.circle(frame, hole, int(3*HOLE_RADIUS_PX), self.danger_color, 2)

class PathDetector:
    def __init__(self):
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 30])
        self.min_line_width = 50
        self.path_points = []

    def detect_path(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > self.min_line_width]
        
        if valid_contours:
            combined = np.vstack(valid_contours)
            sorted_points = combined[np.argsort(combined[:, 0, 0])].squeeze()
            if len(sorted_points) > 4:
                self.path_points = self._create_spline_path(sorted_points)
            
        return mask, self.path_points

    def _create_spline_path(self, points):
        t = np.arange(len(points))
        cs_x = CubicSpline(t, points[:, 0])
        cs_y = CubicSpline(t, points[:, 1])
        t_new = np.linspace(0, len(points)-1, 100)
        return np.column_stack((cs_x(t_new), cs_y(t_new))).astype(int)

# ========== Main Application ==========
def rotate_point(point, angle):
    x, y = point
    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle)
    )

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracker = BallTracker()
    hole_mgr = HoleManager()
    path_detector = PathDetector()
    manual_path = []
    
    cv2.namedWindow("Board Controller")
    cv2.setMouseCallback("Board Controller", lambda *args: manual_path.append((args[1], args[0])))

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Detect path and track ball
        path_mask, auto_path = path_detector.detect_path(frame)
        current_path = manual_path if manual_path else auto_path
        ball_state, velocity = tracker.update(frame)
        
        # Visualization
        hole_mgr.draw_holes(frame)
        if len(current_path) > 1:
            cv2.polylines(frame, [np.array(current_path)], False, (255,0,0), 2)
        
        if ball_state is not None:
            bx, by = int(ball_state[0]), int(ball_state[1])
            cv2.circle(frame, (bx, by), 10, (0,255,0), -1)
            
            if current_path:
                target = current_path[np.argmin(np.linalg.norm(current_path - [bx, by], axis=1))]
                cv2.circle(frame, tuple(target), 8, (0,0,255), 2)
                cv2.line(frame, (bx, by), tuple(target), (255,0,0), 2)
                
                avoid_vec = hole_mgr.check_collision((bx, by), velocity)
                if np.linalg.norm(avoid_vec) > 0:
                    avoid_end = (int(bx + avoid_vec[0]*50), int(by + avoid_vec[1]*50))
                    cv2.arrowedLine(frame, (bx, by), avoid_end, (0,255,255), 2)

        # Show debug views
        debug_mask = cv2.cvtColor(path_mask, cv2.COLOR_GRAY2BGR)
        debug_mask = cv2.resize(debug_mask, (640, 360))
        combined = np.vstack((frame, debug_mask))
        
        cv2.imshow("Board Controller", combined)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
