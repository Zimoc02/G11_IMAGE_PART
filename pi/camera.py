import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import smbus
from collections import deque
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

# ========== System Setup ==========
bus = smbus.SMBus(1)
arduino_addr = 0x08

def rotate_point(point, angle):
    """Rotate point by given angle (radians)"""
    x, y = point
    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle)
    )

# ========== Path Planning ==========
class BoardSpline:
    def __init__(self, waypoints_cm):
        self.waypoints = np.array([(x/PX_TO_CM_X, y/PX_TO_CM_Y) 
                                  for x,y in waypoints_cm])
        self.s = np.arange(len(waypoints_cm))
        self.cs_x = CubicSpline(self.s, self.waypoints[:,0])
        self.cs_y = CubicSpline(self.s, self.waypoints[:,1])
        
    def project(self, point_px):
        """Brent's method projection from thesis Algorithm 2"""
        def f(s):
            dx = point_px[0] - self.cs_x(s)
            dy = point_px[1] - self.cs_y(s)
            tx = self.cs_x.derivative()(s)
            ty = self.cs_y.derivative()(s)
            return tx*dy - ty*dx
        
        try:
            s_root = brentq(f, 0, len(self.s)-1)
            return (self.cs_x(s_root), self.cs_y(s_root))
        except:
            distances = np.linalg.norm(self.waypoints - point_px, axis=1)
            return self.waypoints[np.argmin(distances)]

# ========== Computer Vision ==========
class BallTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4,2)
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
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(largest)
            measurement = np.array([[x], [y]], dtype=np.float32)
            
            # Kalman update
            self.kalman.correct(measurement)
            predicted = self.kalman.predict()
            
            # Velocity calculation (px/frame)
            vel = self.kalman.statePost[2:4].flatten()
            vel_cm = (vel * PX_TO_CM_X, vel * PX_TO_CM_Y)
            
            # Stuck detection
            if np.linalg.norm(vel_cm) < STUCK_VELOCITY_THRESH:
                delta_time = (cv2.getTickCount() - self.last_update)/cv2.getTickFrequency()
                if delta_time > 2.0:  # 2 seconds stuck
                    self.integrated_angle += INTEGRATION_RATE
            else:
                self.integrated_angle = 0.0
                self.last_update = cv2.getTickCount()
            
            return (predicted[0], predicted[1]), vel_cm
        return None, None

# ========== Obstacle Database ==========    
# ========== Obstacle Database ==========    
class HoleManager:
    def __init__(self):
        self.holes = [(535,173),(443,172),(364,180)]  # Populate with hole positions in pixels
        
    def check_collision(self, pos_px, vel_px):
        risk_vector = np.zeros(2)
        pos_px = np.array(pos_px)  # Convert position to numpy array
        for hole in self.holes:
            hole_arr = np.array(hole)  # Convert hole to numpy array
            displacement = pos_px - hole_arr  # Now works with numpy vectors
            dist = np.linalg.norm(displacement)
            direction = displacement / (dist + 1e-6)  # Avoid division by zero
            
            approach_speed = np.dot(vel_px, direction)
            if dist < 3*HOLE_RADIUS_PX and approach_speed < 0:
                alpha = D1*(1 - dist/(3*HOLE_RADIUS_PX)) - D2*abs(approach_speed)
                risk_vector += alpha * direction
        return risk_vector

# ========== Main Loop ==========
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracker = BallTracker()
    hole_mgr = HoleManager()
    spline_path = BoardSpline(waypoints_cm=[(5,5), (15,12), (25,5)])
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Tracking
        ball_state, velocity = tracker.update(frame)
        
        if ball_state is not None:
            ball_pos = (ball_state[0], ball_state[1])
            
            # Path following with anti-stuck
            target_pos = np.array(spline_path.project(ball_pos))
            if tracker.integrated_angle != 0:
                target_pos = rotate_point(target_pos - ball_pos, tracker.integrated_angle) + ball_pos
            
            # Collision avoidance
            avoid_vec = hole_mgr.check_collision(ball_pos, velocity)
            if np.linalg.norm(avoid_vec) > 0:
                target_pos += avoid_vec * HOLE_RADIUS_PX*2
                
            # Send to Arduino
            data = [
                int(ball_pos[0]), int(ball_pos[1]),
                int(target_pos[0]), int(target_pos[1])
            ]
            bus.write_i2c_block_data(arduino_addr, 0, data)
            
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()
