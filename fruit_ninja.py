"""Fruit Ninja with your hand. Swipe your hand through the flying fruit to slice it.

Run from the gesture-control folder:
    .venv/bin/python fruit_ninja.py

You appear in the camera; the blade follows your index fingertip. Swipe fast through
the fruit to slice it (a slow hand won't cut). Let fruit fall and you lose a life
(3 strikes). Slice a bomb and the run ends. Several fruit in one quick swipe scores a
combo bonus.

Keys: q quit, r restart after game over, b toggle camera / dark background.
"""

import collections
import math
import os
import random
import subprocess
import tempfile
import time
import wave
from array import array

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_tracking import MODEL_PATH

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG_DIM = 0.45          # how much to darken the camera feed behind the game
SLICE_SPEED = 550.0    # px/sec the fingertip must exceed to actually cut
BLADE_R = 10           # blade thickness added to a fruit's slice radius
COMBO_WINDOW = 0.35    # seconds; fruit cut this close together count as a combo
BOMB_CHANCE = 0.14
START_LIVES = 3

# name, rind BGR, flesh BGR, radius, leaf
FRUIT_TYPES = [
    ("watermelon", (55, 130, 55),  (70, 70, 215),  46, True),
    ("orange",     (40, 140, 250), (120, 200, 255), 34, True),
    ("apple",      (50, 50, 215),  (225, 240, 245), 32, True),
    ("lemon",      (70, 215, 240), (170, 245, 250), 30, False),
    ("lime",       (90, 200, 120), (175, 240, 200), 27, False),
    ("grape",      (120, 50, 95),  (190, 140, 200), 24, False),
]
BOMB_RADIUS = 32

SR = 22050  # sound sample rate


def scale(color, f):
    return tuple(int(min(255, max(0, c * f))) for c in color)


def blend(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def seg_circle_hit(p1, p2, center, r):
    (x1, y1), (x2, y2) = p1, p2
    cx, cy = center
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return (cx - x1) ** 2 + (cy - y1) ** 2 <= r * r
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / length2))
    px, py = x1 + t * dx, y1 + t * dy
    return (cx - px) ** 2 + (cy - py) ** 2 <= r * r


# --- sound: short effects synthesized once, played with afplay (non-blocking) ---

def _to_wave(samples):
    out = array("h")
    for s in samples:
        out.append(int(max(-1.0, min(1.0, s)) * 32767))
    return out


def _slice_samples():
    n = int(0.20 * SR)
    prev = 0.0
    for i in range(n):
        t = i / SR
        env = math.exp(-t * 16) * min(1.0, t / 0.004)
        sweep = math.sin(2 * math.pi * (1500 - 3800 * t) * t)
        prev = prev * 0.6 + random.uniform(-1, 1) * 0.4
        yield (prev * 0.7 + sweep * 0.3) * env * 0.8


def _bomb_samples():
    n = int(0.6 * SR)
    prev = 0.0
    for i in range(n):
        t = i / SR
        env = math.exp(-t * 5.0)
        base = math.sin(2 * math.pi * (95 - 55 * t) * t)
        prev = prev * 0.85 + random.uniform(-1, 1) * 0.15
        s = (base * 0.7 + prev * 0.5) * env
        if t < 0.01:
            s += random.uniform(-1, 1) * (1 - t / 0.01)
        yield s * 0.9


def _combo_samples():
    n = int(0.22 * SR)
    for i in range(n):
        t = i / SR
        env = math.exp(-t * 9) * min(1.0, t / 0.004)
        f = 720 if t < 0.07 else 1080
        yield math.sin(2 * math.pi * f * t) * env * 0.6


def _over_samples():
    n = int(0.7 * SR)
    for i in range(n):
        t = i / SR
        env = math.exp(-t * 3.5)
        f = 420 - 210 * t
        yield (math.sin(2 * math.pi * f * t) + 0.5 * math.sin(2 * math.pi * f * 2 * t)) * env * 0.5


def _miss_samples():
    n = int(0.18 * SR)
    prev = 0.0
    for i in range(n):
        t = i / SR
        env = math.exp(-t * 14)
        prev = prev * 0.8 + random.uniform(-1, 1) * 0.2
        yield (math.sin(2 * math.pi * 160 * t) * 0.6 + prev * 0.4) * env * 0.5


class Sounds:
    """Synthesizes the effects to temp WAVs once, then fires them with afplay.
    Each play spawns a short non-blocking process so the game loop never waits."""

    GENERATORS = {
        "slice": _slice_samples,
        "bomb": _bomb_samples,
        "combo": _combo_samples,
        "over": _over_samples,
        "miss": _miss_samples,
    }

    def __init__(self):
        self.ok = False
        self.paths = {}
        try:
            folder = tempfile.mkdtemp(prefix="fn_snd_")
            for name, gen in self.GENERATORS.items():
                path = os.path.join(folder, name + ".wav")
                with wave.open(path, "w") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes(_to_wave(gen()).tobytes())
                self.paths[name] = path
            self.ok = True
        except Exception:
            self.ok = False

    def play(self, name):
        if not self.ok or name not in self.paths:
            return
        try:
            subprocess.Popen(["afplay", self.paths[name]],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


class Fruit:
    def __init__(self, x, y, vx, vy, r, color, kind, leaf, flesh=None, name=""):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.r, self.color, self.kind, self.leaf = r, color, kind, leaf
        self.flesh = flesh if flesh is not None else blend(color, (255, 255, 255), 0.55)
        self.name = name
        self.spin = random.uniform(-90, 90)
        self.angle = random.uniform(0, 360)
        self.sliced = False


class Half:
    def __init__(self, x, y, vx, vy, r, color, flesh, angle, spin):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.r, self.color, self.flesh = r, color, flesh
        self.angle, self.spin = angle, spin
        self.life = 1.4


class Particle:
    def __init__(self, x, y, vx, vy, r, color, life):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.r, self.color = r, color
        self.life = self.life0 = life


class Splat:
    """A juice mark left on the background where a fruit was cut."""

    def __init__(self, x, y, color, blobs, life):
        self.x, self.y, self.color, self.blobs = x, y, color, blobs
        self.life = self.life0 = life


class Game:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.reset()

    def reset(self):
        self.fruits = []
        self.halves = []
        self.particles = []
        self.splats = []
        self.events = []
        self.score = 0
        self.lives = START_LIVES
        self.over = False
        self.spawn_timer = 0.6
        self.recent_slices = []
        self.combo_text = ""
        self.combo_until = 0.0
        self.bomb_flash = 0.0
        self.shake = 0.0

    def _spawn_one(self):
        x = random.uniform(self.w * 0.15, self.w * 0.85)
        vx = random.uniform(self.w * 0.05, self.w * 0.18)
        vx = abs(vx) if x < self.w * 0.5 else -abs(vx)  # aim back toward center
        vy = -random.uniform(self.h * 1.15, self.h * 1.4)  # upward launch
        if random.random() < BOMB_CHANCE:
            self.fruits.append(
                Fruit(x, self.h + BOMB_RADIUS, vx, vy, BOMB_RADIUS,
                      (45, 45, 48), "bomb", False, name="bomb"))
        else:
            name, color, flesh, r, leaf = random.choice(FRUIT_TYPES)
            self.fruits.append(
                Fruit(x, self.h + r, vx, vy, r, color, "fruit", leaf,
                      flesh=flesh, name=name))

    def _spawn_wave(self):
        for _ in range(random.randint(1, 3)):
            self._spawn_one()

    def _spawn_halves(self, f):
        for sign in (-1, 1):
            self.halves.append(Half(
                f.x, f.y,
                f.vx + sign * random.uniform(60, 150),
                f.vy - random.uniform(20, 90),
                f.r, f.color, f.flesh,
                random.uniform(0, 180), sign * random.uniform(120, 260)))

    def _spawn_particles(self, x, y, color, n, spread=260):
        for _ in range(n):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(40, spread)
            self.particles.append(Particle(
                x, y, math.cos(ang) * spd, math.sin(ang) * spd,
                random.randint(3, 8), color, random.uniform(0.35, 0.7)))

    def _spawn_splat(self, x, y, color, r):
        blobs = [(0.0, 0.0, r * 0.55)]
        for _ in range(6):
            ang = random.uniform(0, 2 * math.pi)
            d = random.uniform(0.3, 1.2) * r
            blobs.append((math.cos(ang) * d, math.sin(ang) * d,
                          random.uniform(0.15, 0.4) * r))
        self.splats.append(Splat(x, y, color, blobs, life=1.2))
        if len(self.splats) > 26:
            self.splats = self.splats[-26:]

    def _slice(self, f, now):
        f.sliced = True
        if f.kind == "bomb":
            self._spawn_particles(f.x, f.y, (0, 140, 255), 30, spread=440)
            self._spawn_particles(f.x, f.y, (60, 60, 60), 16, spread=300)
            self.over = True
            self.bomb_flash = 1.0
            self.shake = 1.0
            self.events.append("bomb")
            return
        self.score += 1
        self._spawn_halves(f)
        self._spawn_particles(f.x, f.y, f.flesh, 18)
        self._spawn_splat(f.x, f.y, f.flesh, f.r)
        self.events.append("slice")
        self.recent_slices = [t for t in self.recent_slices if now - t < COMBO_WINDOW]
        self.recent_slices.append(now)
        if len(self.recent_slices) >= 2:
            bonus = len(self.recent_slices)
            self.score += bonus
            self.combo_text = f"Combo x{len(self.recent_slices)}   +{bonus}"
            self.combo_until = now + 1.2
            self.events.append("combo")

    def update(self, dt, now, seg, speed, blade_r=BLADE_R):
        self.events = []
        self.bomb_flash = max(0.0, self.bomb_flash - dt * 2.0)
        self.shake = max(0.0, self.shake - dt * 3.0)

        g = self.h * 1.1
        for p in self.particles:
            p.vy += g * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]

        for hf in self.halves:
            hf.vy += g * dt
            hf.x += hf.vx * dt
            hf.y += hf.vy * dt
            hf.angle += hf.spin * dt
            hf.life -= dt
        self.halves = [h for h in self.halves if h.life > 0 and h.y < self.h + 250]

        for s in self.splats:
            s.life -= dt
        self.splats = [s for s in self.splats if s.life > 0]

        if self.over:
            return

        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self._spawn_wave()
            interval = max(0.55, 1.3 - self.score * 0.01)
            self.spawn_timer = interval * random.uniform(0.7, 1.2)

        for f in self.fruits:
            f.vy += g * dt
            f.x += f.vx * dt
            f.y += f.vy * dt
            f.angle += f.spin * dt

        if seg is not None and speed > SLICE_SPEED:
            for f in self.fruits:
                if not f.sliced and seg_circle_hit(seg[0], seg[1], (f.x, f.y), f.r + blade_r):
                    self._slice(f, now)

        survivors = []
        for f in self.fruits:
            if f.sliced:
                continue
            if f.y > self.h + f.r * 2 and f.vy > 0:
                if f.kind == "fruit":
                    self.lives -= 1
                    self.events.append("miss")
                    if self.lives <= 0:
                        self.over = True
                        self.events.append("over")
                continue
            survivors.append(f)
        self.fruits = survivors


def draw_sphere(frame, cx, cy, r, base):
    # radial shade (bright center, dark rim) reads as a lit ball; cheap and fast.
    for rr in range(r, 0, -2):
        shade = 1.15 - 0.6 * (rr / r)
        cv2.circle(frame, (cx, cy), rr, scale(base, shade), -1)
    cv2.circle(frame, (cx - int(r * 0.32), cy - int(r * 0.32)),
               max(2, r // 4), blend(base, (255, 255, 255), 0.6), -1)
    cv2.circle(frame, (cx, cy), r, scale(base, 0.5), 2)


def draw_fruit(frame, f):
    cx, cy, r = int(f.x), int(f.y), f.r
    draw_sphere(frame, cx, cy, r, f.color)
    if f.name == "watermelon":
        for k in (-1, 0, 1):
            ox = int(k * r * 0.4)
            hh = int(math.sqrt(max(0, r * r - ox * ox)))
            cv2.ellipse(frame, (cx + ox, cy), (max(2, r // 10), hh), 0, 0, 360,
                        scale(f.color, 0.55), -1)
    elif f.name == "apple":
        cv2.line(frame, (cx, cy - r), (cx + int(r * 0.12), cy - int(r * 1.3)),
                 (40, 60, 95), 3)
    if f.leaf:
        cv2.ellipse(frame, (cx + int(r * 0.25), cy - int(r * 0.95)),
                    (max(3, r // 4), max(2, r // 9)), -35, 0, 360, (60, 150, 70), -1)


def draw_bomb(frame, b):
    cx, cy, r = int(b.x), int(b.y), b.r
    draw_sphere(frame, cx, cy, r, (45, 45, 48))
    cv2.circle(frame, (cx, cy), r, (0, 0, 0), 2)
    cv2.ellipse(frame, (cx, cy), (r, int(r * 0.5)), 0, 0, 360, (20, 20, 20), 2)
    fx, fy = cx, cy - r
    cv2.line(frame, (fx, fy), (fx + 10, fy - 15), (60, 80, 110), 3)
    spark = 4 + random.randint(0, 3)
    cv2.circle(frame, (fx + 10, fy - 15), spark, (0, 200, 255), -1)
    cv2.circle(frame, (fx + 10, fy - 15), max(1, spark // 2), (255, 255, 255), -1)


def draw_half(frame, h):
    c = (int(h.x), int(h.y))
    cv2.ellipse(frame, c, (h.r, h.r), h.angle, 0, 180, h.flesh, -1)
    cv2.ellipse(frame, c, (int(h.r * 0.66), int(h.r * 0.66)), h.angle, 0, 180,
                blend(h.flesh, (255, 255, 255), 0.3), -1)
    cv2.ellipse(frame, c, (h.r, h.r), h.angle, 0, 180, h.color, max(3, h.r // 6))


def draw_particles(frame, particles):
    for p in particles:
        rad = max(1, int(p.r * p.life / p.life0))
        cv2.circle(frame, (int(p.x), int(p.y)), rad, p.color, -1)


def draw_splats(frame, splats):
    if not splats:
        return
    overlay = frame.copy()
    for s in splats:
        f = s.life / s.life0
        for dx, dy, rr in s.blobs:
            cv2.circle(overlay, (int(s.x + dx), int(s.y + dy)),
                       max(1, int(rr * f)), s.color, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


def draw_trail(frame, points, width=BLADE_R):
    if len(points) < 2:
        return
    width = min(width, 34)  # visual blade width; the cut radius can run wider
    overlay = frame.copy()
    for i in range(1, len(points)):
        thick = max(3, int((width + 8) * (0.35 + 0.65 * i / len(points))))
        cv2.line(overlay, points[i - 1], points[i], (255, 235, 190), thick)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    for i in range(1, len(points)):
        thick = max(1, int(width * 0.5 * i / len(points)))
        cv2.line(frame, points[i - 1], points[i], (255, 255, 255), thick)
    cv2.circle(frame, points[-1], max(5, width // 3), (255, 255, 255), -1)


def draw_strike(frame, cx, cy, s, color, th):
    cv2.line(frame, (cx - s, cy - s), (cx + s, cy + s), color, th)
    cv2.line(frame, (cx - s, cy + s), (cx + s, cy - s), color, th)


def draw_hud(frame, game, now):
    cv2.putText(frame, str(game.score), (20, 56), FONT, 1.6, (0, 0, 0), 7)
    cv2.putText(frame, str(game.score), (20, 56), FONT, 1.6, (255, 255, 255), 2)
    width = frame.shape[1]
    used = START_LIVES - game.lives
    for i in range(START_LIVES):
        cx = width - 36 - i * 46
        if i < used:
            draw_strike(frame, cx, 36, 13, (40, 40, 235), 5)
        else:
            draw_strike(frame, cx, 36, 13, (110, 110, 110), 3)
    if now < game.combo_until and game.combo_text:
        size = cv2.getTextSize(game.combo_text, FONT, 1.5, 4)[0]
        x = (width - size[0]) // 2
        cv2.putText(frame, game.combo_text, (x, 124), FONT, 1.5, (0, 0, 0), 8)
        cv2.putText(frame, game.combo_text, (x, 124), FONT, 1.5, (40, 230, 255), 4)


def draw_game_over(frame, game, best):
    h, w = frame.shape[:2]
    frame[:] = cv2.convertScaleAbs(frame, alpha=0.3)
    pw, ph = int(w * 0.6), int(h * 0.6)
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (30, 25, 22), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (60, 180, 240), 3)
    lines = [("GAME OVER", 1.7, (60, 60, 235), 4),
             (f"Score   {game.score}", 1.1, (255, 255, 255), 2),
             (f"Best   {best}", 0.9, (120, 220, 255), 2),
             ("r = play again      q = quit", 0.7, (200, 200, 200), 2)]
    y = y0 + 75
    for text, sc, color, th in lines:
        size = cv2.getTextSize(text, FONT, sc, th)[0]
        cv2.putText(frame, text, ((w - size[0]) // 2, y), FONT, sc, (0, 0, 0), th + 3)
        cv2.putText(frame, text, ((w - size[0]) // 2, y), FONT, sc, color, th)
        y += size[1] + 32


def make_dark_bg(w, h):
    # warm dark gradient with a vignette, so the dark mode looks like a dojo wall.
    bg = np.zeros((h, w, 3), np.float32)
    top = np.array((58, 48, 40), np.float32)
    bot = np.array((26, 22, 20), np.float32)
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    bg[:] = top * (1 - ramp) + bot * ramp
    yy, xx = np.ogrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    v = np.clip(1.0 - 0.45 * d, 0.4, 1.0).astype(np.float32)
    return (bg * v[..., None]).astype(np.uint8)


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
    sounds = Sounds()
    if not sounds.ok:
        print("Sound effects unavailable; running silent.")

    game = None
    best = 0
    dark_bg = None
    trail = collections.deque(maxlen=14)
    bg_camera = True
    start = prev = time.time()
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        if game is None:
            game = Game(width, height)
            dark_bg = make_dark_bg(width, height)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        now = time.time()
        dt = min(now - prev, 0.05)
        prev = now

        seg, speed, blade_r = None, 0.0, BLADE_R
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            # The blade tracks the index fingertip, the leading point of a swipe,
            # so it follows where you point. Sized to the hand so a closer hand
            # (bigger on screen) gets a slightly wider blade.
            tip = lm[8]
            point = (int(tip.x * width), int(tip.y * height))
            span = math.hypot((lm[9].x - lm[0].x) * width, (lm[9].y - lm[0].y) * height)
            blade_r = max(BLADE_R, min(34, int(span * 0.45)))
            trail.append(point)
            if len(trail) >= 2 and dt > 0:
                p0, p1 = trail[-2], trail[-1]
                speed = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt
                seg = (p0, p1)
        else:
            trail.clear()

        game.update(dt, now, seg, speed, blade_r)
        if game.over:
            best = max(best, game.score)

        for name in ("bomb", "over", "combo", "miss", "slice"):
            if name in game.events:
                sounds.play(name)

        if bg_camera:
            frame = cv2.convertScaleAbs(frame, alpha=BG_DIM)
        else:
            frame = dark_bg.copy()

        draw_splats(frame, game.splats)
        draw_particles(frame, game.particles)
        for h in game.halves:
            draw_half(frame, h)
        for f in game.fruits:
            if f.kind == "bomb":
                draw_bomb(frame, f)
            else:
                draw_fruit(frame, f)
        draw_trail(frame, list(trail), blade_r)
        draw_hud(frame, game, now)

        if now - start < 3.0:
            msg = "Swipe to slice the fruit. Avoid the bombs!"
            size = cv2.getTextSize(msg, FONT, 0.8, 2)[0]
            x = (width - size[0]) // 2
            cv2.putText(frame, msg, (x, height // 2), FONT, 0.8, (0, 0, 0), 5)
            cv2.putText(frame, msg, (x, height // 2), FONT, 0.8, (255, 255, 255), 2)

        if game.over:
            draw_game_over(frame, game, best)

        if game.bomb_flash > 0.001:
            a = min(0.7, game.bomb_flash * 0.7)
            solid = np.full(frame.shape, (60, 150, 255), np.uint8)
            cv2.addWeighted(solid, a, frame, 1 - a, 0, frame)

        if game.shake > 0.001:
            mag = game.shake * 16
            M = np.float32([[1, 0, random.uniform(-mag, mag)],
                            [0, 1, random.uniform(-mag, mag)]])
            frame = cv2.warpAffine(frame, M, (width, height),
                                   borderMode=cv2.BORDER_REPLICATE)

        cv2.imshow("Fruit Ninja", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and game.over:
            game.reset()
            start = time.time()
        if key == ord("b"):
            bg_camera = not bg_camera

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
