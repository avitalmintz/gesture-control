"""Webcam hand tracking with MediaPipe Tasks (HandLandmarker).

Run from the gesture-control folder:
    .venv/bin/python hand_tracking.py

Press q to quit. The first run triggers a macOS camera permission prompt
for your terminal; approve it and rerun if the window closes.
"""

import math
import subprocess
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import Quartz
    from AppKit import NSEvent
    MEDIA_KEYS_OK = True
except ImportError:
    MEDIA_KEYS_OK = False

MODEL_PATH = "hand_landmarker.task"

# A pointing finger sweeps this horizontal band (mirrored frame x, 0-1) across the
# full 0-100 volume range. Margins so the very edges of the frame stay reachable.
VOL_X_LO = 0.15
VOL_X_HI = 0.85
VOL_SMOOTH = 0.35  # easing toward the target; lower is smoother but laggier

NX_KEYTYPE_PLAY = 16   # macOS play/pause media key
PALM_DWELL = 0.4       # seconds an open hand must be held before it fires play/pause

# 21-point hand skeleton. Drawn by hand because the Tasks API has no built-in overlay.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(landmarks):
    wrist = landmarks[0]
    pinky_mcp = landmarks[17]
    up = []
    # A digit is extended when its tip is farther from a reference joint than the
    # joint below it. Fingers reach out from the wrist; the thumb swings sideways
    # from the pinky's base. Using distances (not raw x/y) survives a tilted hand.
    up.append(_dist(landmarks[4], pinky_mcp) > _dist(landmarks[3], pinky_mcp))
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        up.append(_dist(landmarks[tip], wrist) > _dist(landmarks[pip], wrist))
    return up


def name_gesture(up):
    thumb, index, middle, ring, pinky = up
    count = sum(up)
    if count == 0:
        return "Fist"
    if all(up):
        return "Open palm"
    if index and middle and not ring and not pinky and not thumb:
        return "Peace"
    if index and not middle and not ring and not pinky and not thumb:
        return "Pointing"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs up"
    return f"{count} fingers"


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def draw_volume_bar(frame, vol, active):
    height, width = frame.shape[:2]
    x0, x1 = 20, width - 20
    y0, y1 = height - 60, height - 40
    color = (0, 255, 0) if active else (200, 200, 200)
    fill = int(x0 + (x1 - x0) * vol / 100)
    cv2.rectangle(frame, (x0, y0), (fill, y1), color, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(frame, f"Vol {int(round(vol))}  (point and slide)", (x0, y0 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def send_play_pause():
    """Post the system play/pause media key, toggling whatever is playing
    (browser video, Spotify, Music, etc.). Needs Accessibility permission for
    the terminal running this, or macOS silently drops the event."""
    if not MEDIA_KEYS_OK:
        return
    for down in (True, False):
        flags = 0xA00 if down else 0xB00
        data1 = (NX_KEYTYPE_PLAY << 16) | ((0xA if down else 0xB) << 8)
        event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            14, (0, 0), flags, 0, 0, None, 8, data1, -1)
        Quartz.CGEventPost(0, event.CGEvent())


class VolumeController:
    """Applies macOS output volume from a background thread so the camera loop
    never blocks on osascript (each call spawns a process, ~20ms)."""

    def __init__(self):
        self._target = self._read_volume()
        self._applied = self._target
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _read_volume():
        try:
            out = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True, timeout=2,
            )
            return int(out.stdout.strip())
        except (ValueError, subprocess.SubprocessError):
            return 50

    def set_target(self, value):
        with self._lock:
            self._target = max(0, min(100, int(round(value))))

    @property
    def value(self):
        with self._lock:
            return self._target

    def _run(self):
        while not self._stop:
            target = self.value
            if target != self._applied:
                try:
                    subprocess.run(
                        ["osascript", "-e", f"set volume output volume {target}"],
                        timeout=2,
                    )
                except subprocess.SubprocessError:
                    pass
                self._applied = target
            time.sleep(0.05)

    def close(self):
        self._stop = True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        print("On macOS: System Settings > Privacy & Security > Camera, enable")
        print("your terminal, then run this again.")
        return

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    volume = VolumeController()
    level = float(volume.value)

    if MEDIA_KEYS_OK:
        print("Hold an open hand = play/pause. If nothing happens, grant your")
        print("terminal Accessibility access: System Settings > Privacy &")
        print("Security > Accessibility, enable it, then rerun.")
    else:
        print("Media keys unavailable; open-hand play/pause is off.")

    prev = time.time()
    last_ts = -1
    palm_since = -1.0
    palm_fired = False
    last_pause = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires strictly increasing millisecond timestamps.
        ts = int(time.time() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        height, width = frame.shape[:2]
        controlling = False
        palm_seen = False
        for landmarks in result.hand_landmarks:
            draw_hand(frame, landmarks)
            up = fingers_up(landmarks)
            gesture = name_gesture(up)
            x, y = int(landmarks[0].x * width), int(landmarks[0].y * height)
            cv2.putText(frame, gesture, (x - 20, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Index alone (a pointing hand) grabs the slider; its tip's horizontal
            # position sets the level. Thumb state is ignored so the grip is forgiving.
            pointing = up[1] and not up[2] and not up[3] and not up[4]
            if pointing and not controlling:
                span = (landmarks[8].x - VOL_X_LO) / (VOL_X_HI - VOL_X_LO)
                level += (max(0.0, min(100.0, span * 100)) - level) * VOL_SMOOTH
                volume.set_target(level)
                controlling = True

            # All four fingers up (an open hand) is the play/pause pose.
            if up[1] and up[2] and up[3] and up[4]:
                palm_seen = True

        draw_volume_bar(frame, level, controlling)

        now = time.time()
        # Require a brief hold so an open hand in passing (e.g. releasing the volume
        # slider) does not toggle playback. Fire once, then wait for the hand to drop.
        if palm_seen:
            if palm_since < 0:
                palm_since = now
            elif not palm_fired and now - palm_since >= PALM_DWELL:
                send_play_pause()
                palm_fired = True
                last_pause = now
        else:
            palm_since = -1.0
            palm_fired = False
        if now - last_pause < 0.6:
            cv2.putText(frame, "PLAY/PAUSE", (width - 235, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:4.0f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "press q to quit", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    volume.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
