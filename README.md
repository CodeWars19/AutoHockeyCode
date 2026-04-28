# Air Hockey Robot — Vision System

## Files

| File | Purpose |
|---|---|
| `air_hockey_vision.py` | Main vision + control loop (run on Raspberry Pi) |
| `calibrate_hsv.py` | Interactive HSV tuner for your puck color |
| `air_hockey_robot.ino` | Arduino sketch (motor driver) |

---

## Quick Start

### 1. Install dependencies on Raspberry Pi

```bash
pip install opencv-python numpy pyserial
```

### 2. Calibrate your puck color

```bash
python calibrate_hsv.py --camera 0
```

- Drag sliders until **only your puck** appears white in the mask.
- Press **S** to save `hsv_values.json`.
- Copy the printed `RED_LOWER1/UPPER1/2` values into `air_hockey_vision.py`.

### 3. Configure `air_hockey_vision.py`

Edit the **CONFIGURATION** block at the top:

```python
ARDUINO_PORT = "/dev/ttyUSB0"  # find with: ls /dev/tty*
CAMERA_INDEX = 0               # try 0, 1, 2 if wrong camera opens
FRAME_W      = 1280
FRAME_H      = 720
```

### 4. Calibrate table boundaries

Run the main script:
```bash
python air_hockey_vision.py
```

- Press **C** to enter calibration mode.
- Click the **4 corners** of the table playing surface.
- The terminal prints `TABLE_LEFT/RIGHT/TOP/BOTTOM` — copy these into the config.

### 5. Flash the Arduino

- Open `air_hockey_robot.ino` in Arduino IDE.
- Set `USE_SERVOS` to `true` (servo) or `false` (stepper) depending on your hardware.
- Adjust pin numbers to match your wiring.
- Flash to your Arduino.

### 6. Tune robot IK

In `air_hockey_vision.py`, adjust:

```python
L1 = 180.0   # your actual link lengths (any unit, just be consistent)
L2 = 180.0
L3 = 180.0
L4 = 180.0
BASE_SEP = 120.0  # distance between motor pivots
```

Also adjust `pixel_to_robot_frame()` to map table pixels → robot coordinates correctly.

---

## Architecture

```
DV128 Camera
     │  frames @ 60fps
     ▼
┌─────────────────────────────────────────────────┐
│  HSV Color Filter (red puck)                    │
│  Contour + circularity detection                │
│  Lucas-Kanade Optical Flow (velocity)           │
│  Kalman Filter (smooth pos + vel)               │
│  ↓                                              │
│  Physics Trajectory Prediction                  │
│    - ray-cast to walls, apply bounce physics    │
│    - up to 3 wall bounces predicted             │
│  ↓                                              │
│  Intercept Point Selection                      │
│    - walk trajectory, check robot can reach     │
│    - pick furthest feasible intercept           │
│  ↓                                              │
│  5-Bar Linkage Inverse Kinematics               │
│    - solve θ1, θ2 for target (x, y)             │
│  ↓                                              │
│  Serial → Arduino → Motor PWM/Steps             │
└─────────────────────────────────────────────────┘
```

## Velocity Estimation Strategy

Two complementary methods are blended:

1. **Lucas-Kanade Optical Flow** — tracks the puck pixel-by-pixel between frames. Fast, responsive, works even if the color filter misses a frame.

2. **Kalman Filter** — 4-state `[x, y, vx, vy]` filter. Smooths out detection jitter and predicts position when the puck is briefly occluded.

These are blended 55% optical-flow / 45% Kalman, which you can tune with the `blend` variable.

## Tuning Tips

| Problem | Fix |
|---|---|
| Puck not detected | Run `calibrate_hsv.py`, re-tune HSV |
| Robot overshoots | Lower `ROBOT_SPEED_PX_PER_S` |
| Robot too slow to react | Raise `ROBOT_SPEED_PX_PER_S`, lower `TIME_BUFFER_S` |
| Jittery robot | Increase Kalman process noise, decrease blend toward OF |
| Wrong bounce prediction | Tune `BOUNCE_PERP` / `BOUNCE_PARA` coefficients |
| Serial not found | `ls /dev/tty*` before/after plugging in Arduino |

## Keys (while running)

| Key | Action |
|---|---|
| Q | Quit |
| C | Calibration mode (click 4 table corners) |
| R | Reset calibration clicks |
| H | Send robot home |
