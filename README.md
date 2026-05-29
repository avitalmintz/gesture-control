# Gesture Control

Control your Mac and play with your hands using just a webcam. Built on
[MediaPipe](https://developers.google.com/mediapipe) hand tracking and OpenCV.
The same 21-point hand skeleton drives four separate apps: a volume and
media controller, an air mouse, a Fruit Ninja game, and an air-paint canvas.

Built and tested on macOS (Apple Silicon). Some features use macOS-only APIs
(system volume, cursor events, media keys, sound playback).

## What's included

| File | What it does |
| --- | --- |
| `hand_tracking.py` | Point and slide to set system volume; hold an open hand to play/pause media. Shows the hand skeleton and the recognized gesture. |
| `air_mouse.py` | Move the macOS cursor with your index finger. Pinch to click, hold the pinch and move to drag. |
| `fruit_ninja.py` | Fruit Ninja with your hand. The blade follows your fingertip; swipe fast to slice fruit, avoid the bombs. Sound effects, combos, and a juice-splatter look. |
| `air_paint.py` | Paint in the air. One finger draws, two fingers move and pick tools. Ten colors, four sizes, and five brushes (Pen, Neon, Spray, Rainbow, Calligraphy). |

The MediaPipe model (`hand_landmarker.task`) is included, so nothing extra to
download.

## Setup

```bash
git clone https://github.com/avitalmintz/gesture-control.git
cd gesture-control
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running

Run any app with the project's Python:

```bash
.venv/bin/python hand_tracking.py
.venv/bin/python air_mouse.py
.venv/bin/python fruit_ninja.py
.venv/bin/python air_paint.py
```

### hand_tracking.py
Point with one finger and slide left/right to set the volume. Hold an open
hand (all fingers up) for a moment to send play/pause to whatever is playing.
Press `q` to quit.

### air_mouse.py
Point your index finger to move the cursor inside the on-screen box. Pinch
thumb and index together to click; hold the pinch and move to drag. Press `q`
to quit.

### fruit_ninja.py
Swipe your hand fast through the flying fruit to slice it (a slow hand will not
cut). Avoid the bombs, and do not let fruit fall past the bottom (three lives).
Keys: `q` quit, `r` restart after game over, `b` toggle camera or dark
background.

### air_paint.py
Point with one finger to paint. Raise a second finger (index and middle) to
stop painting and move around; hold that two-finger pointer over a tool in the
top bar for about half a second to pick it. Keys: `b` cycle background (camera,
white, black), `c` clear, `e` toggle eraser, `q` quit.

## macOS permissions

- **Camera:** every app needs it. The first run prompts your terminal under
  System Settings > Privacy & Security > Camera.
- **Accessibility:** `air_mouse.py` (cursor events) and the play/pause key in
  `hand_tracking.py` need it. Enable your terminal under System Settings >
  Privacy & Security > Accessibility, then rerun. Without it, macOS silently
  drops those events.
