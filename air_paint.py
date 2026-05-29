"""Air paint: draw in the air with your hand, pinch to paint and to pick tools.

Run from the gesture-control folder:
    .venv/bin/python air_paint.py

Your index fingertip is the brush. Point with one finger to paint. Raise a second
finger (index + middle) to stop painting and move around; hold that two-finger
pointer over a tool in the top bar to pick it: a color, a brush size, a brush type
(Pen, Neon, Spray, Rainbow, Calligraphy), the eraser, or Clear.

Keys: b cycle background (camera / white / black), c clear, e toggle eraser,
q quit.
"""

import collections
import colorsys
import math
import random
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_tracking import MODEL_PATH, fingers_up

FONT = cv2.FONT_HERSHEY_SIMPLEX
TB_H = 84               # toolbar height in pixels
SMOOTH = 0.5            # fingertip easing; lower is steadier but laggier
DWELL = 0.45            # seconds to hold the two-finger pointer on a tool to pick it
CALLI_MAX_SPEED = 2400  # px/sec at which a calligraphy stroke thins to its minimum

# BGR palette shown left-to-right in the toolbar.
PALETTE = [
    (40, 40, 235),    # red
    (40, 140, 250),   # orange
    (40, 220, 250),   # yellow
    (60, 200, 70),    # green
    (200, 200, 40),   # teal
    (235, 120, 40),   # blue
    (200, 60, 160),   # purple
    (170, 120, 250),  # pink
    (245, 245, 245),  # white
    (25, 25, 25),     # black
]
SIZES = [4, 8, 16, 28]
BRUSHES = ["Pen", "Neon", "Spray", "Rainbow", "Calligraphy"]
SPRAY_ICON = [(-8, -4), (-3, 4), (2, -6), (6, 2), (0, 0), (-6, 6), (7, -3)]


def scale(color, f):
    return tuple(int(min(255, max(0, c * f))) for c in color)


def blend(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def hue_bgr(h):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def _lerp_points(p0, p1, step):
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(1, int(dist / step))
    return [(int(p0[0] + (p1[0] - p0[0]) * i / n),
             int(p0[1] + (p1[1] - p0[1]) * i / n)) for i in range(n + 1)]


class Painter:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.canvas = np.zeros((height, width, 3), np.uint8)
        self.mask = np.zeros((height, width), np.uint8)
        self.color = PALETTE[0]
        self.size = SIZES[2]
        self.brush = "Pen"
        self.eraser = False
        self.hue = 0.0

    def clear(self):
        self.canvas[:] = 0
        self.mask[:] = 0

    def select(self, button):
        kind = button["kind"]
        if kind == "color":
            self.color = button["value"]
            self.eraser = False
        elif kind == "size":
            self.size = button["value"]
        elif kind == "brush":
            self.brush = button["value"]
            self.eraser = False
        elif kind == "eraser":
            self.eraser = not self.eraser
        elif kind == "clear":
            self.clear()

    def display_color(self):
        if self.eraser:
            return (255, 255, 255)
        if self.brush == "Rainbow":
            return hue_bgr(self.hue)
        return self.color

    def cursor_radius(self):
        return int(self.size * 2.2) if self.brush == "Neon" else self.size

    def stamp(self, p0, p1, speed=0.0):
        pts = _lerp_points(p0, p1, max(2, self.size * 0.5))
        if self.eraser:
            for x, y in pts:
                cv2.circle(self.mask, (x, y), self.size + 5, 0, -1)
                cv2.circle(self.canvas, (x, y), self.size + 5, (0, 0, 0), -1)
            return
        if self.brush == "Neon":
            glow = scale(self.color, 0.45)
            wide = int(self.size * 2.2)
            for x, y in pts:
                cv2.circle(self.canvas, (x, y), wide, glow, -1)
                cv2.circle(self.mask, (x, y), wide, 255, -1)
            for x, y in pts:
                cv2.circle(self.canvas, (x, y), self.size,
                           blend(self.color, (255, 255, 255), 0.4), -1)
                cv2.circle(self.canvas, (x, y), max(1, self.size // 2), (255, 255, 255), -1)
            return
        for x, y in pts:
            if self.brush == "Spray":
                for _ in range(max(6, self.size * 2)):
                    rx, ry = random.uniform(-1, 1), random.uniform(-1, 1)
                    if rx * rx + ry * ry > 1:
                        continue
                    sx = int(x + rx * self.size * 1.8)
                    sy = int(y + ry * self.size * 1.8)
                    cv2.circle(self.canvas, (sx, sy), 1, self.color, -1)
                    cv2.circle(self.mask, (sx, sy), 1, 255, -1)
            elif self.brush == "Rainbow":
                col = hue_bgr(self.hue)
                self.hue += 0.02
                cv2.circle(self.canvas, (x, y), self.size, col, -1)
                cv2.circle(self.mask, (x, y), self.size, 255, -1)
            elif self.brush == "Calligraphy":
                f = max(0.25, 1.0 - speed / CALLI_MAX_SPEED)
                r = max(1, int(self.size * f))
                cv2.circle(self.canvas, (x, y), r, self.color, -1)
                cv2.circle(self.mask, (x, y), r, 255, -1)
            else:  # Pen
                cv2.circle(self.canvas, (x, y), self.size, self.color, -1)
                cv2.circle(self.mask, (x, y), self.size, 255, -1)


def build_buttons(width, tb_h):
    cy = tb_h // 2
    r = int(tb_h * 0.28)
    btns = []

    def place(fx0, fx1, n, i):
        return int((fx0 + (i + 0.5) * (fx1 - fx0) / n) * width)

    for i, col in enumerate(PALETTE):
        cx = place(0.008, 0.40, len(PALETTE), i)
        btns.append(dict(kind="color", value=col, cx=cx, cy=cy, r=r,
                         x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r))
    for i, sz in enumerate(SIZES):
        cx = place(0.41, 0.52, len(SIZES), i)
        btns.append(dict(kind="size", value=sz, cx=cx, cy=cy, r=r,
                         x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r))
    half = int((0.83 - 0.53) * width / len(BRUSHES) * 0.45)
    for i, b in enumerate(BRUSHES):
        cx = place(0.53, 0.83, len(BRUSHES), i)
        btns.append(dict(kind="brush", value=b, cx=cx, cy=cy, r=r,
                         x0=cx - half, y0=cy - r, x1=cx + half, y1=cy + r))
    aw = int((0.15 * width / 2) * 0.45)
    for i, a in enumerate(["eraser", "clear"]):
        cx = place(0.84, 0.99, 2, i)
        btns.append(dict(kind=a, value=a, cx=cx, cy=cy, r=r,
                         x0=cx - aw, y0=cy - r, x1=cx + aw, y1=cy + r))
    return btns


def hit_test(buttons, x, y):
    for b in buttons:
        if b["x0"] <= x <= b["x1"] and b["y0"] <= y <= b["y1"]:
            return b
    return None


def _draw_brush_icon(frame, name, cx, cy):
    w = (230, 230, 230)
    if name == "Pen":
        cv2.line(frame, (cx - 9, cy + 8), (cx + 7, cy - 8), w, 4)
        cv2.circle(frame, (cx + 8, cy - 9), 2, w, -1)
    elif name == "Neon":
        cv2.circle(frame, (cx, cy), 9, (180, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
    elif name == "Spray":
        for dx, dy in SPRAY_ICON:
            cv2.circle(frame, (cx + dx, cy + dy), 1, w, -1)
    elif name == "Rainbow":
        cv2.ellipse(frame, (cx, cy + 5), (11, 11), 0, 180, 360, (40, 40, 235), 2)
        cv2.ellipse(frame, (cx, cy + 5), (7, 7), 0, 180, 360, (60, 200, 70), 2)
        cv2.ellipse(frame, (cx, cy + 5), (4, 4), 0, 180, 360, (235, 150, 40), 2)
    elif name == "Calligraphy":
        pts = np.array([(cx - 9, cy - 7), (cx + 9, cy + 7), (cx - 9, cy + 7)], np.int32)
        cv2.fillPoly(frame, [pts], w)


def draw_button(frame, b, painter):
    kind = b["kind"]
    cx, cy, r = b["cx"], b["cy"], b["r"]
    cyan = (60, 220, 255)
    if kind == "color":
        cv2.circle(frame, (cx, cy), r, b["value"], -1)
        cv2.circle(frame, (cx, cy), r, (90, 90, 90), 1)
        if not painter.eraser and painter.color == b["value"]:
            cv2.circle(frame, (cx, cy), r + 3, cyan, 3)
    elif kind == "size":
        cv2.circle(frame, (cx, cy), min(b["value"], r - 3), (210, 210, 210), -1)
        if painter.size == b["value"]:
            cv2.circle(frame, (cx, cy), r + 2, cyan, 3)
    else:
        selected = ((kind == "brush" and not painter.eraser and painter.brush == b["value"])
                    or (kind == "eraser" and painter.eraser))
        bg = (70, 70, 82) if selected else (48, 48, 56)
        cv2.rectangle(frame, (b["x0"], b["y0"]), (b["x1"], b["y1"]), bg, -1)
        cv2.rectangle(frame, (b["x0"], b["y0"]), (b["x1"], b["y1"]),
                      cyan if selected else (95, 95, 105), 2 if selected else 1)
        if kind == "brush":
            _draw_brush_icon(frame, b["value"], cx, cy)
        elif kind == "eraser":
            cv2.rectangle(frame, (cx - 9, cy - 6), (cx + 9, cy + 7), (170, 120, 250), -1)
            cv2.rectangle(frame, (cx - 9, cy - 6), (cx + 9, cy + 7), (240, 240, 240), 1)
        else:  # clear
            cv2.rectangle(frame, (cx - 7, cy - 3), (cx + 7, cy + 10), (230, 230, 230), 2)
            cv2.line(frame, (cx - 10, cy - 5), (cx + 10, cy - 5), (230, 230, 230), 2)
            cv2.line(frame, (cx - 3, cy - 8), (cx + 3, cy - 8), (230, 230, 230), 2)


def draw_toolbar(frame, buttons, painter):
    width = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, TB_H), (30, 30, 35), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, TB_H), (width, TB_H), (80, 80, 90), 2)
    for b in buttons:
        draw_button(frame, b, painter)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    painter = None
    buttons = None
    white_bg = black_bg = None
    bg_mode = 0  # 0 camera, 1 white, 2 black
    cx = cy = None
    prev = None       # last painted point while a stroke is down
    last = None       # last cursor point, for speed
    dwell_btn = None
    dwell_start = 0.0
    dwell_done = False
    mode = "idle"
    start = prev_t = time.time()
    last_ts = -1
    print("Point one finger to paint. Two fingers to move and pick tools. q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        if painter is None:
            painter = Painter(width, height)
            buttons = build_buttons(width, TB_H)
            white_bg = np.full((height, width, 3), 245, np.uint8)
            black_bg = np.full((height, width, 3), 18, np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        now = time.time()
        dt = max(now - prev_t, 1e-6)
        prev_t = now

        speed = 0.0
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            tip = lm[8]
            tx, ty = tip.x * width, tip.y * height
            if cx is None:
                cx, cy = tx, ty
            else:
                cx += (tx - cx) * SMOOTH
                cy += (ty - cy) * SMOOTH
            cur = (int(cx), int(cy))
            if last is not None:
                speed = math.hypot(cur[0] - last[0], cur[1] - last[1]) / dt
            last = cur

            # The middle finger is the toggle: index alone paints, index + middle
            # lifts the pen so you can move and pick tools without drawing a line.
            up = fingers_up(lm)
            if up[1] and not up[2]:
                mode = "paint"
                dwell_btn = None
                dwell_done = False
                if cur[1] >= TB_H:
                    if prev is not None:
                        painter.stamp(prev, cur, speed)
                    prev = cur
                else:
                    prev = None
            elif up[1] and up[2]:
                mode = "move"
                prev = None
                btn = hit_test(buttons, cur[0], cur[1]) if cur[1] < TB_H else None
                if btn is None:
                    dwell_btn = None
                    dwell_done = False
                elif btn is not dwell_btn:
                    dwell_btn = btn
                    dwell_start = now
                    dwell_done = False
                elif not dwell_done and now - dwell_start >= DWELL:
                    painter.select(btn)
                    dwell_done = True
            else:
                mode = "idle"
                prev = None
                dwell_btn = None
                dwell_done = False
        else:
            cx = cy = last = None
            mode = "idle"
            prev = None
            dwell_btn = None
            dwell_done = False

        if bg_mode == 0:
            out = cv2.convertScaleAbs(frame, alpha=0.55)
        elif bg_mode == 1:
            out = white_bg.copy()
        else:
            out = black_bg.copy()
        cv2.copyTo(painter.canvas, painter.mask, out)

        draw_toolbar(out, buttons, painter)

        if cx is not None:
            ix, iy = int(cx), int(cy)
            disp = painter.display_color()
            rr = max(3, painter.cursor_radius())
            if mode == "paint":
                cv2.circle(out, (ix, iy), rr, (0, 0, 0), 3)
                cv2.circle(out, (ix, iy), rr, disp, 2)
                if iy >= TB_H:
                    cv2.circle(out, (ix, iy), max(2, rr - 3), disp, -1)
            elif mode == "move":
                cv2.circle(out, (ix, iy), 12, (0, 0, 0), 3)
                cv2.circle(out, (ix, iy), 12, (255, 255, 255), 2)
                cv2.circle(out, (ix, iy), 3, (255, 255, 255), -1)
                if dwell_btn is not None and not dwell_done:
                    prog = max(0.0, min(1.0, (now - dwell_start) / DWELL))
                    cv2.ellipse(out, (ix, iy), (18, 18), 0, -90, -90 + int(360 * prog),
                                (60, 220, 255), 3)
                elif dwell_done:
                    cv2.circle(out, (ix, iy), 18, (60, 220, 255), 3)
            else:
                cv2.circle(out, (ix, iy), 5, (180, 180, 180), 2)

        mode_txt = {"paint": "Paint", "move": "Move", "idle": "-"}[mode]
        label = f"{mode_txt}    " + ("Eraser" if painter.eraser else painter.brush) + f"    size {painter.size}"
        cv2.putText(out, label, (16, height - 20), FONT, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (16, height - 20), FONT, 0.7, (255, 255, 255), 2)
        if now - start < 7.0:
            msg = "One finger = paint.   Two fingers = move + hold over a tool to pick."
            size = cv2.getTextSize(msg, FONT, 0.7, 2)[0]
            x = (width - size[0]) // 2
            cv2.putText(out, msg, (x, TB_H + 36), FONT, 0.7, (0, 0, 0), 5)
            cv2.putText(out, msg, (x, TB_H + 36), FONT, 0.7, (255, 255, 255), 2)

        cv2.imshow("Air Paint", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            painter.clear()
        if key == ord("e"):
            painter.eraser = not painter.eraser
        if key == ord("b"):
            bg_mode = (bg_mode + 1) % 3

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
