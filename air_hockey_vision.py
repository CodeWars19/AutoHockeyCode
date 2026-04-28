"""
Air Hockey Robot - Vision & Control System
==========================================
Raspberry Pi + Arduino + DV128 top-down camera

Pipeline:
  1. Capture frame from DV128 camera
  2. HSV color filter (red puck + green table border)
  3. Optical-flow / frame-differencing for velocity estimation
  4. Kalman filter for smoothed position + velocity
  5. Physics-based trajectory prediction (wall bounces)
  6. 5-linkage robot IK solver → motor commands via Serial

Wiring assumption:
  - Arduino on /dev/ttyUSB0 (change ARDUINO_PORT below)
  - Camera index 0 (change CAMERA_INDEX if needed)

Color conventions:
  - Puck  : RED (use a red puck, or change HSV ranges)
  - Table border/mallet: GREEN (optional, used for boundary calibration)
"""

import cv2
import numpy as np
import serial
import time
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these to match your physical setup
# ─────────────────────────────────────────────────────────────────────────────

ARDUINO_PORT   = "/dev/ttyUSB0"   # Linux: /dev/ttyUSB0  |  Mac: /dev/cu.usbmodem...
BAUD_RATE      = 115200
CAMERA_INDEX   = 0                # 0 = first camera; change if DV128 is index 1/2
FRAME_W        = 1280
FRAME_H        = 720
CAMERA_FPS     = 60               # request 60 fps from camera

# Table pixel boundaries (calibrate with the debug window)
TABLE_LEFT   = 66
TABLE_RIGHT  = 1222
TABLE_TOP    = 80
TABLE_BOTTOM = 640

# Robot home position in pixel-space (centre of robot's reachable arc)
ROBOT_HOME_X = 101   # robot sits near left edge
ROBOT_HOME_Y = 360   # vertically centred

# Robot reachable x range (pixels) — robot defends left side
ROBOT_MIN_X  = 60
ROBOT_MAX_X  = 600   # don't let robot cross half-table

# Motor/robot speed estimate (pixels per second) — tune after testing
ROBOT_SPEED_PX_PER_S = 2200

# Time buffer: extra seconds robot needs as safety margin
TIME_BUFFER_S = 0.12

# Puck detection: HSV range for RED puck
# Red in HSV wraps around 0/180 — we use two ranges and OR them
RED_LOWER1 = np.array([0,   120,  80])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([165, 120,  80])
RED_UPPER2 = np.array([180, 255, 255])

# Green filter for optional table/mallet detection
GREEN_LOWER = np.array([35, 50, 30])
GREEN_UPPER = np.array([90, 255, 255])

# Minimum contour area to count as puck (filters noise)
PUCK_MIN_AREA = 150
PUCK_MAX_AREA = 8000

# Physics — coefficient of restitution for wall bounces
BOUNCE_PERP  = 0.72   # velocity component perpendicular to wall
BOUNCE_PARA  = 0.85   # velocity component parallel to wall

# Minimum speed (px/s) to bother predicting — below this, treat as stationary
MIN_SPEED_TO_PREDICT = 120

# ─────────────────────────────────────────────────────────────────────────────
# KALMAN FILTER — position + velocity tracking
# ─────────────────────────────────────────────────────────────────────────────

class PuckKalman:
    """
    4-state Kalman filter: [x, y, vx, vy]
    Measurement: [x, y]

    Uses standard OpenCV KalmanFilter internally.
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state dims, 2 measurement dims

        dt = 1.0 / CAMERA_FPS

        # State transition: x' = x + vx*dt, y' = y + vy*dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # Measurement: we observe x, y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise — how much we trust the motion model
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] = 5.0   # higher uncertainty on velocity
        self.kf.processNoiseCov[3, 3] = 5.0

        # Measurement noise — camera detection uncertainty (~3 px)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 9.0

        # Initial error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        self.initialized = False

    def update(self, x: float, y: float) -> Tuple[float, float, float, float]:
        """Feed a new measurement; returns smoothed (x, y, vx, vy)."""
        meas = np.array([[x], [y]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True

        self.kf.predict()
        state = self.kf.correct(meas)
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])

    def predict_only(self) -> Tuple[float, float, float, float]:
        """Predict next position without a measurement (puck occluded)."""
        state = self.kf.predict()
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])

    def set_dt(self, dt: float):
        """Update transition matrix with current frame delta-time."""
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt


# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY PREDICTION (frame differencing + physics)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    x: float
    y: float
    wall: str   # 't','b','l','r','n'

def ray_wall_intersect(
    x0: float, y0: float, vx: float, vy: float
) -> TrajectoryPoint:
    """
    Cast a ray from (x0,y0) with direction (vx,vy).
    Returns the first wall intersection and which wall was hit.
    """
    left   = TABLE_LEFT
    right  = TABLE_RIGHT
    top    = TABLE_TOP
    bottom = TABLE_BOTTOM

    t_min = float('inf')
    wall = 'n'

    if vx > 0:
        t = (right - x0) / vx
        if 0 < t < t_min and top <= y0 + t * vy <= bottom:
            t_min, wall = t, 'r'
    elif vx < 0:
        t = (left - x0) / vx
        if 0 < t < t_min and top <= y0 + t * vy <= bottom:
            t_min, wall = t, 'l'

    if vy > 0:
        t = (bottom - y0) / vy
        if 0 < t < t_min and left <= x0 + t * vx <= right:
            t_min, wall = t, 'b'
    elif vy < 0:
        t = (top - y0) / vy
        if 0 < t < t_min and left <= x0 + t * vx <= right:
            t_min, wall = t, 't'

    if t_min == float('inf'):
        return TrajectoryPoint(x0, y0, 'n')

    xf = x0 + t_min * vx
    yf = y0 + t_min * vy
    return TrajectoryPoint(xf, yf, wall)


def predict_trajectory(
    x: float, y: float, vx: float, vy: float, max_bounces: int = 3
) -> list:
    """
    Simulate puck path with wall bounces.
    Returns list of TrajectoryPoint (including starting position).
    """
    points = [TrajectoryPoint(x, y, 'n')]
    cx, cy, cvx, cvy = x, y, vx, vy

    for _ in range(max_bounces):
        pt = ray_wall_intersect(cx, cy, cvx, cvy)
        if pt.wall == 'n':
            break
        points.append(pt)

        # Apply bounce physics
        if pt.wall in ('t', 'b'):
            cvx = cvx * BOUNCE_PARA
            cvy = -cvy * BOUNCE_PERP
        else:
            cvx = -cvx * BOUNCE_PERP
            cvy = cvy * BOUNCE_PARA

        cx, cy = pt.x, pt.y

        # Stop if speed too low to matter
        if abs(cvx) + abs(cvy) < MIN_SPEED_TO_PREDICT * 0.3:
            break

    return points


def draw_trajectory(frame: np.ndarray, points: list):
    """Draw predicted path on debug frame."""
    colors = [(0, 200, 200), (50, 150, 255), (150, 255, 50)]
    for i in range(len(points) - 1):
        p0 = (int(points[i].x),   int(points[i].y))
        p1 = (int(points[i+1].x), int(points[i+1].y))
        color = colors[min(i, len(colors)-1)]
        cv2.line(frame, p0, p1, color, 2)
        cv2.circle(frame, p1, 5, color, -1)


# ─────────────────────────────────────────────────────────────────────────────
# PUCK DETECTION (HSV color filter + contour)
# ─────────────────────────────────────────────────────────────────────────────

def detect_puck(frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Detect the red puck using HSV masking + contour finding.
    Returns (cx, cy, radius) or None if not found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around hue=0, so OR two masks
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup — remove noise, fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < PUCK_MIN_AREA or area > PUCK_MAX_AREA:
            continue

        # Circularity check — puck is a circle
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < 0.5:
            continue

        # Score = area * circularity (prefer big, round blobs)
        score = area * circularity
        if score > best_score:
            best_score = score
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            best = (x, y, radius)

    return best


def detect_green_mask(frame: np.ndarray) -> np.ndarray:
    """Optional: detect green table elements for boundary calibration."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 5-LINKAGE INVERSE KINEMATICS (simplified)
# ─────────────────────────────────────────────────────────────────────────────
# Adjust these lengths to match your robot's physical linkage dimensions (mm or px)

L1 = 180.0   # proximal link length (motor 1 side)
L2 = 180.0   # distal link length  (motor 1 side)
L3 = 180.0   # proximal link length (motor 2 side)
L4 = 180.0   # distal link length  (motor 2 side)
BASE_SEP = 120.0  # distance between the two motor pivots

# Robot base pivot positions (in robot's local coordinate frame, px)
BASE1 = np.array([0.0,       0.0])
BASE2 = np.array([BASE_SEP,  0.0])

def ik_5bar(target_x: float, target_y: float) -> Optional[Tuple[float, float]]:
    """
    Closed-form IK for symmetric 5-bar linkage.
    Returns (theta1_deg, theta2_deg) for the two motors, or None if unreachable.

    target_x, target_y: end-effector position in robot local frame (origin = BASE1)
    """
    # IK for arm 1: reach from BASE1 to target
    def ik_2link(base, L_a, L_b, px, py):
        dx, dy = px - base[0], py - base[1]
        r = math.hypot(dx, dy)
        if r > L_a + L_b or r < abs(L_a - L_b) or r == 0:
            return None
        cos_elbow = (r**2 + L_a**2 - L_b**2) / (2 * r * L_a)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        alpha = math.acos(cos_elbow)
        base_angle = math.atan2(dy, dx)
        theta = base_angle + alpha   # elbow-up solution
        return math.degrees(theta)

    theta1 = ik_2link(BASE1, L1, L2, target_x, target_y)
    theta2 = ik_2link(BASE2, L3, L4, target_x, target_y)

    if theta1 is None or theta2 is None:
        return None

    return (theta1, theta2)


def pixel_to_robot_frame(px_x: float, px_y: float) -> Tuple[float, float]:
    """
    Convert pixel-space coordinates to robot local frame (mm or px units).
    Adjust the scale and origin to match your physical setup.

    Assumption: robot pivot BASE1 maps to (ROBOT_HOME_X, ROBOT_HOME_Y) in pixel space,
    and the robot arm works in a coordinate system where +y is towards the table centre.
    """
    # Scale factor: pixels per unit (adjust after measuring your table)
    PX_PER_UNIT = 1.0   # if your table coords are already in the same units as L1/L2

    robot_x = (px_x - ROBOT_HOME_X) * PX_PER_UNIT
    robot_y = (px_y - ROBOT_HOME_Y) * PX_PER_UNIT

    return robot_x, robot_y


# ─────────────────────────────────────────────────────────────────────────────
# ROBOT CONTROL LOGIC — intercept point selection
# ─────────────────────────────────────────────────────────────────────────────

def find_intercept(
    traj_points: list,
    vx: float, vy: float,
    puck_x: float, puck_y: float,
    prev_cmd: Tuple[float, float]
) -> Tuple[float, float, bool]:
    """
    Walk along predicted trajectory to find the best intercept point.
    Returns (intercept_x, intercept_y, should_move).
    """
    if abs(vx) + abs(vy) < MIN_SPEED_TO_PREDICT:
        # Puck nearly stationary — drive to it if it's in our half
        if puck_x < (TABLE_LEFT + TABLE_RIGHT) / 2:
            return puck_x, puck_y, True
        return ROBOT_HOME_X, ROBOT_HOME_Y, False

    # Only try to intercept if puck is moving toward robot (negative vx)
    if vx >= 0:
        return ROBOT_HOME_X, ROBOT_HOME_Y, False

    # Compute cumulative time along trajectory segments
    speed = math.hypot(vx, vy)
    cvx, cvy = vx, vy

    best_ix, best_iy = ROBOT_HOME_X, ROBOT_HOME_Y
    best_found = False

    # Sample every 8 pixels along the trajectory
    for seg_idx in range(len(traj_points) - 1):
        p0 = traj_points[seg_idx]
        p1 = traj_points[seg_idx + 1]

        seg_len = math.hypot(p1.x - p0.x, p1.y - p0.y)
        if seg_len == 0:
            continue

        steps = max(1, int(seg_len / 8))
        for step in range(steps + 1):
            t = step / steps
            ix = p0.x + t * (p1.x - p0.x)
            iy = p0.y + t * (p1.y - p0.y)

            # Only consider robot's half of the table
            if ix > (TABLE_LEFT + TABLE_RIGHT) / 2:
                continue
            if ix < ROBOT_MIN_X or ix > ROBOT_MAX_X:
                continue

            # Time for puck to reach this point
            dist_puck = math.hypot(ix - puck_x, iy - puck_y)
            t_puck = dist_puck / max(speed, 1.0)

            # Time for robot to reach this point
            dist_robot = math.hypot(ix - prev_cmd[0], iy - prev_cmd[1])
            t_robot = dist_robot / ROBOT_SPEED_PX_PER_S

            if t_robot + TIME_BUFFER_S <= t_puck:
                best_ix, best_iy = ix, iy
                best_found = True
                break   # take the first (closest to puck) viable point

        if best_found:
            break

    return best_ix, best_iy, best_found


# ─────────────────────────────────────────────────────────────────────────────
# ARDUINO COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────────

class ArduinoComm:
    def __init__(self, port: str, baud: int):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(1.5)   # wait for Arduino reset
            self.connected = True
            print(f"[Arduino] Connected on {port} @ {baud}")
        except serial.SerialException as e:
            print(f"[Arduino] WARNING: Could not connect — {e}")
            print("[Arduino] Running in simulation mode (no serial output)")
            self.connected = False

    def send_angles(self, theta1: float, theta2: float):
        """Send two motor angles (degrees) to Arduino."""
        if not self.connected:
            return
        # Protocol: "T1:{angle1},T2:{angle2}\n"
        msg = f"T1:{theta1:.1f},T2:{theta2:.1f}\n"
        try:
            self.ser.write(msg.encode())
        except serial.SerialException:
            pass

    def send_home(self):
        """Send robot to home/default position."""
        self.send_angles(90.0, 90.0)

    def close(self):
        if self.connected:
            self.ser.close()


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION HELPER — find table corners via mouse click
# ─────────────────────────────────────────────────────────────────────────────

calib_points = []

def mouse_callback(event, x, y, flags, param):
    global calib_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calib_points.append((x, y))
        print(f"[Calibration] Clicked: ({x}, {y})  |  total={len(calib_points)}")
        if len(calib_points) == 4:
            xs = [p[0] for p in calib_points]
            ys = [p[1] for p in calib_points]
            print(f"[Calibration] TABLE_LEFT={min(xs)}, TABLE_RIGHT={max(xs)}, "
                  f"TABLE_TOP={min(ys)}, TABLE_BOTTOM={max(ys)}")


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL FLOW — matrix-differenced frame integration for velocity
# ─────────────────────────────────────────────────────────────────────────────

class OpticalFlowVelocity:
    """
    Compute dense optical flow between consecutive frames to derive
    puck velocity independent of detection jitter.
    Uses Lucas-Kanade sparse tracking on the detected puck centre.
    """
    def __init__(self):
        self.prev_gray  = None
        self.prev_point = None
        self.lk_params  = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

    def update(
        self,
        gray: np.ndarray,
        detected_x: float,
        detected_y: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Returns (vx, vy) in pixels/second using optical flow when available,
        falling back to 0,0 on first frame.
        """
        vx, vy = 0.0, 0.0

        if self.prev_gray is not None and self.prev_point is not None:
            p0 = np.array([[self.prev_point]], dtype=np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None, **self.lk_params
            )
            if st[0][0] == 1 and dt > 0:
                dx = float(p1[0][0][0]) - float(p0[0][0][0])
                dy = float(p1[0][0][1]) - float(p0[0][0][1])
                vx = dx / dt
                vy = dy / dt

        self.prev_gray  = gray.copy()
        self.prev_point = (detected_x, detected_y)
        return vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Air Hockey Robot — Vision & Control")
    print("=" * 60)
    print("  Keys:")
    print("    Q — quit")
    print("    C — enter calibration mode (click 4 table corners)")
    print("    R — reset calibration")
    print("    H — send robot HOME")
    print("=" * 60)

    # ── Camera setup ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)  # use CAP_DSHOW on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # minimize latency

    if not cap.isOpened():
        print("[Camera] ERROR: Cannot open camera. Check CAMERA_INDEX.")
        sys.exit(1)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] Running at {actual_fps:.0f} fps")

    # ── Arduino ───────────────────────────────────────────────────────────
    arduino = ArduinoComm(ARDUINO_PORT, BAUD_RATE)

    # ── Kalman + optical flow ─────────────────────────────────────────────
    kalman = PuckKalman()
    of_tracker = OpticalFlowVelocity()

    # ── State ─────────────────────────────────────────────────────────────
    prev_cmd  = (ROBOT_HOME_X, ROBOT_HOME_Y)
    prev_time = time.perf_counter()
    missing_frames = 0
    MAX_MISSING = 10   # frames without detection before giving up

    # Position history for HUD trail
    history = deque(maxlen=40)

    calibration_mode = False

    # ── Windows ───────────────────────────────────────────────────────────
    cv2.namedWindow("Air Hockey — Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Hockey — Debug", 1280, 720)
    cv2.setMouseCallback("Air Hockey — Debug", mouse_callback)

    print("[System] Running. Press Q to quit.")

    while True:
        now = time.perf_counter()
        dt  = now - prev_time
        prev_time = now

        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame read failed — retrying...")
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug = frame.copy()

        # ── Draw table boundaries ─────────────────────────────────────────
        cv2.rectangle(debug,
                      (TABLE_LEFT, TABLE_TOP),
                      (TABLE_RIGHT, TABLE_BOTTOM),
                      (0, 255, 0), 2)

        # ── Puck detection ────────────────────────────────────────────────
        detection = detect_puck(frame)

        if detection is not None:
            raw_x, raw_y, radius = detection
            missing_frames = 0

            # Optical flow velocity (pixels/s)
            of_vx, of_vy = of_tracker.update(gray, raw_x, raw_y, dt)

            # Kalman update — smooth position
            kalman.set_dt(dt)
            kx, ky, kvx, kvy = kalman.update(raw_x, raw_y)

            # Blend optical-flow velocity with Kalman velocity estimate
            # (OF is more responsive; Kalman is smoother)
            blend = 0.55
            vx = blend * of_vx + (1 - blend) * kvx
            vy = blend * of_vy + (1 - blend) * kvy

            # Draw raw detection
            cv2.circle(debug, (int(raw_x), int(raw_y)), int(radius), (0, 0, 255), 2)
            # Draw Kalman position
            cv2.circle(debug, (int(kx), int(ky)), 6, (255, 100, 0), -1)

            # Velocity arrow
            arrow_scale = 0.08
            ax = int(kx + vx * arrow_scale)
            ay = int(ky + vy * arrow_scale)
            cv2.arrowedLine(debug, (int(kx), int(ky)), (ax, ay), (0, 255, 255), 2, tipLength=0.3)

            # History trail
            history.append((int(kx), int(ky)))

        else:
            missing_frames += 1
            of_tracker.update(gray, *prev_cmd, dt)

            if missing_frames < MAX_MISSING and kalman.initialized:
                kalman.set_dt(dt)
                kx, ky, kvx, kvy = kalman.predict_only()
                vx, vy = kvx, kvy
            else:
                kx, ky, vx, vy = ROBOT_HOME_X, ROBOT_HOME_Y, 0.0, 0.0

        # Draw position trail
        for i in range(1, len(history)):
            alpha = i / len(history)
            color = (int(50 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.line(debug, history[i-1], history[i], color, 2)

        # ── Trajectory prediction ─────────────────────────────────────────
        speed = math.hypot(vx, vy)
        traj  = []

        if speed > MIN_SPEED_TO_PREDICT:
            traj = predict_trajectory(kx, ky, vx, vy, max_bounces=3)
            draw_trajectory(debug, traj)

        # ── Intercept & robot command ─────────────────────────────────────
        ix, iy, should_move = find_intercept(traj, vx, vy, kx, ky, prev_cmd)

        if not should_move:
            ix, iy = ROBOT_HOME_X, ROBOT_HOME_Y

        prev_cmd = (ix, iy)

        # Draw intercept target
        cv2.circle(debug, (int(ix), int(iy)), 14, (0, 50, 255), 3)
        cv2.circle(debug, (int(ix), int(iy)), 4,  (0, 50, 255), -1)

        # Draw robot position + line to intercept
        cv2.circle(debug, (ROBOT_HOME_X, ROBOT_HOME_Y), 10, (255, 255, 255), 2)
        cv2.line(debug,
                 (ROBOT_HOME_X, ROBOT_HOME_Y),
                 (int(ix), int(iy)),
                 (180, 180, 180), 1)

        # ── IK → motor angles ─────────────────────────────────────────────
        robot_x, robot_y = pixel_to_robot_frame(ix, iy)
        ik_result = ik_5bar(robot_x, robot_y)

        if ik_result is not None:
            theta1, theta2 = ik_result
            arduino.send_angles(theta1, theta2)

            cv2.putText(debug,
                        f"θ1={theta1:.1f}° θ2={theta2:.1f}°",
                        (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            arduino.send_home()
            cv2.putText(debug, "IK: unreachable — HOME",
                        (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)

        # ── HUD ───────────────────────────────────────────────────────────
        fps_display = 1.0 / dt if dt > 0 else 0
        cv2.putText(debug, f"FPS: {fps_display:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug, f"Puck: ({kx:.0f}, {ky:.0f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug, f"Speed: {speed:.0f} px/s",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug, f"Vel: ({vx:.0f}, {vy:.0f})",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.putText(debug, f"Target: ({ix:.0f}, {iy:.0f})",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        status = "INTERCEPT" if should_move else "HOME"
        color  = (0, 50, 255) if should_move else (0, 200, 0)
        cv2.putText(debug, f"Mode: {status}",
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if missing_frames > 0:
            cv2.putText(debug, f"LOST PUCK ({missing_frames}f)",
                        (FRAME_W // 2 - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if calibration_mode:
            cv2.putText(debug, "CALIBRATION: click 4 table corners",
                        (FRAME_W // 2 - 200, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            for pt in calib_points:
                cv2.circle(debug, pt, 8, (0, 255, 255), -1)

        cv2.imshow("Air Hockey — Debug", debug)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            calib_points.clear()
            print("[Calibration] Click the 4 corners of the table in the debug window.")
        elif key == ord('r'):
            calib_points.clear()
            print("[Calibration] Reset.")
        elif key == ord('h'):
            arduino.send_home()
            print("[Robot] Sent HOME command.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    arduino.send_home()
    arduino.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()
