"""
HSV Calibration Tool — Air Hockey Puck Tuner
=============================================
Run this script FIRST to find the correct HSV ranges for your puck color.

Controls:
  - Drag sliders to adjust HSV range
  - The right panel shows what the filter sees (white = detected)
  - Press S to save the values, Q to quit

Usage:
  python calibrate_hsv.py
  python calibrate_hsv.py --camera 1   (if your camera is not index 0)
"""

import cv2
import numpy as np
import argparse
import json

WINDOW = "HSV Calibrator"
MASK_W = "Mask Preview"

def nothing(x):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_W,  cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 900, 200)

    # Trackbars — start tuned for red puck
    cv2.createTrackbar("H Low 1",  WINDOW,   0, 180, nothing)
    cv2.createTrackbar("H High 1", WINDOW,  12, 180, nothing)
    cv2.createTrackbar("H Low 2",  WINDOW, 160, 180, nothing)
    cv2.createTrackbar("H High 2", WINDOW, 180, 180, nothing)
    cv2.createTrackbar("S Low",    WINDOW, 110, 255, nothing)
    cv2.createTrackbar("S High",   WINDOW, 255, 255, nothing)
    cv2.createTrackbar("V Low",    WINDOW,  70, 255, nothing)
    cv2.createTrackbar("V High",   WINDOW, 255, 255, nothing)

    print("Adjust sliders until the puck appears WHITE on the mask.")
    print("Press S to save, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hl1 = cv2.getTrackbarPos("H Low 1",  WINDOW)
        hh1 = cv2.getTrackbarPos("H High 1", WINDOW)
        hl2 = cv2.getTrackbarPos("H Low 2",  WINDOW)
        hh2 = cv2.getTrackbarPos("H High 2", WINDOW)
        sl  = cv2.getTrackbarPos("S Low",    WINDOW)
        sh  = cv2.getTrackbarPos("S High",   WINDOW)
        vl  = cv2.getTrackbarPos("V Low",    WINDOW)
        vh  = cv2.getTrackbarPos("V High",   WINDOW)

        lower1 = np.array([hl1, sl, vl])
        upper1 = np.array([hh1, sh, vh])
        lower2 = np.array([hl2, sl, vl])
        upper2 = np.array([hh2, sh, vh])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Draw contours on frame
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                cv2.circle(display, (int(x), int(y)), int(r), (0, 255, 0), 3)
                cv2.putText(display, f"A={area:.0f}", (int(x)-20, int(y)-int(r)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        side = np.hstack([display, result])
        cv2.imshow(MASK_W, side)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            vals = {
                "RED_LOWER1": [hl1, sl, vl],
                "RED_UPPER1": [hh1, sh, vh],
                "RED_LOWER2": [hl2, sl, vl],
                "RED_UPPER2": [hh2, sh, vh],
            }
            with open("hsv_values.json", "w") as f:
                json.dump(vals, f, indent=2)
            print("\n[Saved] hsv_values.json:")
            print(json.dumps(vals, indent=2))
            print("\nCopy these into air_hockey_vision.py:")
            print(f"  RED_LOWER1 = np.array({vals['RED_LOWER1']})")
            print(f"  RED_UPPER1 = np.array({vals['RED_UPPER1']})")
            print(f"  RED_LOWER2 = np.array({vals['RED_LOWER2']})")
            print(f"  RED_UPPER2 = np.array({vals['RED_UPPER2']})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
