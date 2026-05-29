"""Air mouse: move the macOS cursor with your index finger, pinch to click.

Run from the gesture-control folder:
    .venv/bin/python air_mouse.py

Point with your index finger to move the cursor inside the on-screen box.
Pinch thumb and index together to click; hold the pinch and move to drag.
Press q in the camera window (or Ctrl+C in the terminal) to quit.

Posting cursor events needs Accessibility permission: System Settings >
Privacy & Security > Accessibility, enable your terminal, then rerun.
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import Quartz

from hand_tracking import MODEL_PATH, draw_hand, _dist

# A centered camera box maps to the whole screen, so the edges are reachable
# without your hand leaving the frame.
MARGIN = 0.15
CURSOR_SMOOTH = 0.4   # cursor easing; lower is steadier but laggier

# Pinch = thumb-tip to index-tip distance scaled by palm width, with hysteresis
# (engage tighter than release) so it does not flicker between click and release.
PINCH_ON = 0.45
PINCH_OFF = 0.70


def screen_size():
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return float(bounds.size.width), float(bounds.size.height)


def post_mouse(event_type, x, y):
    event = Quartz.CGEventCreateMouseEvent(
        None, event_type, (x, y), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def to_screen(nx, ny, sw, sh):
    span = 1.0 - 2.0 * MARGIN
    fx = min(max((nx - MARGIN) / span, 0.0), 1.0)
    fy = min(max((ny - MARGIN) / span, 0.0), 1.0)
    return fx * sw, fy * sh


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    sw, sh = screen_size()
    print(f"Screen mapped: {int(sw)} x {int(sh)} points.")
    print("Pinch thumb + index to click, hold to drag. q or Ctrl+C to quit.")
    print("If the cursor does not move, grant your terminal Accessibility access")
    print("in System Settings > Privacy & Security > Accessibility, then rerun.")

    cx = cy = None
    pinching = False
    prev = time.time()
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int(time.time() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        height, width = frame.shape[:2]
        rx0, ry0 = int(MARGIN * width), int(MARGIN * height)
        rx1, ry1 = int((1 - MARGIN) * width), int((1 - MARGIN) * height)
        cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (80, 80, 80), 1)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_hand(frame, landmarks)

            tip = landmarks[8]
            tx, ty = to_screen(tip.x, tip.y, sw, sh)
            if cx is None:
                cx, cy = tx, ty
            else:
                cx += (tx - cx) * CURSOR_SMOOTH
                cy += (ty - cy) * CURSOR_SMOOTH

            palm = _dist(landmarks[5], landmarks[17]) or 1e-6
            ratio = _dist(landmarks[4], landmarks[8]) / palm

            if not pinching and ratio < PINCH_ON:
                pinching = True
                post_mouse(Quartz.kCGEventLeftMouseDown, cx, cy)
            elif pinching and ratio > PINCH_OFF:
                pinching = False
                post_mouse(Quartz.kCGEventLeftMouseUp, cx, cy)
            else:
                moved = (Quartz.kCGEventLeftMouseDragged if pinching
                         else Quartz.kCGEventMouseMoved)
                post_mouse(moved, cx, cy)

            fx, fy = int(tip.x * width), int(tip.y * height)
            color = (0, 0, 255) if pinching else (0, 255, 255)
            cv2.circle(frame, (fx, fy), 12, color, -1 if pinching else 2)
        elif pinching:
            # Hand vanished mid-pinch; release so the button is not left stuck down.
            pinching = False
            if cx is not None:
                post_mouse(Quartz.kCGEventLeftMouseUp, cx, cy)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:4.0f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "pinch = click / drag    q = quit", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Air Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
