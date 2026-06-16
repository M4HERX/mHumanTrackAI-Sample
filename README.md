# mHumanTrackAI Hand Gesture Sample

Turn 21 hand points from a webcam into simple gestures like a pinch, a fist, or an open
palm. This is a tiny, standalone piece of [mHumanTrackAI](https://mhumantrackai.m4herx.com),
shared here so you can read the core idea on its own and try it in a few seconds.

> **Live demo and full app:** **https://mhumantrackai.m4herx.com**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

---

## What this is

mHumanTrackAI watches you through an ordinary webcam and turns your hand and body
movement into mouse, keyboard, and game controller input. The webcam side gives you a
set of points for each hand. The interesting part, the part that decides "this is a
pinch, that is a fist", is just geometry.

This sample pulls that geometry out into one short, readable file with no camera and no
machine learning model attached. It runs anywhere Python runs.

What it can tell from a hand:

* which fingers are up or curled
* is the hand a **fist** or an **open palm**
* are two fingertips **pinched** together
* one short label per hand, ready to map to an action like a left click

## How a hand becomes 21 points

A webcam tracker such as MediaPipe Hands gives you 21 points for each hand. Each point
is an `(x, y)` pair from 0 to 1 across the image. Point 0 is the wrist, and the fingertips
are points 4, 8, 12, 16, and 20.

```
        8   12  16
        |   |   |   20
        7   11  15  |
        |   |   |   19
        6   10  14  |
     4  |   |   |   18
     |  5   9   13  17
     3   \  |   |  /
     2     \ | | /
      \      \|/
       1 ----- 0  (wrist)
```

This sample works on a plain list of 21 `(x, y)` pairs in that order, so it does not
care where the points came from.

## Try it in 30 seconds

You only need Python 3.9 or newer. There is nothing to install.

```bash
# see the gesture reader work on a few built in poses
python demo.py

# run the tests (plain Python, no pytest needed)
python test_hand_gestures.py
```

Example output from the demo:

```
pose given : index pinch
fingers up : {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True}
gesture    : IndexPinch
action     : left click
```

## Try it on your webcam, no setup needed

If you just want to point a real camera at it, download the ready to run app from the
[Releases section](../../releases) of this repository. It is a single Windows file with
Python, the camera libraries, and the hand model already bundled inside, so there is
nothing to install. Open it, hold up a hand, and watch the gesture it reads in real time.
Press **Q**, the **Escape** key, or the window's close button to quit.

Want to run the same live demo from source instead? With `mediapipe` and `opencv-python`
installed, run `python webcam_demo.py`.

## The whole API

```python
import hand_gestures as hg

points = [...]  # a list of 21 (x, y) pairs for one hand

hg.fingers_up(points)        # {'thumb': True, 'index': False, ...}
hg.is_fist(states)           # True when the four main fingers are curled
hg.is_open_palm(states)      # True when every finger is spread out
hg.pinch_distance(points, "thumb", "index")  # 0 means touching, bigger means apart

gesture = hg.detect_gesture(points)     # "IndexPinch", "Fist", "OpenPalm", ...
action  = hg.gesture_to_action(gesture) # "left click", "grab / hold", ...
```

Two small ideas make it reliable:

1. **Scale free.** Every distance is divided by the hand size, measured from the wrist
   to the base of the middle finger. So a pinch reads the same whether you sit close to
   the camera or far away.
2. **A pinch needs a pointing finger.** A fingertip touching the thumb only counts as a
   pinch when that finger is actually pointing out. That is what keeps a fist, where all
   the tips bunch up near the thumb, from being read as a pinch by mistake.

## Plug in a real webcam

The sample stays camera free on purpose, but wiring it to a live camera is short. Install
`mediapipe` and `opencv-python`, then convert each detected hand into the list of pairs
this code expects:

```python
import cv2
import mediapipe as mp
import hand_gestures as hg

hands = mp.solutions.hands.Hands(max_num_hands=1)
camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    if not ok:
        break

    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        points = [(p.x, p.y) for p in landmarks]   # 21 (x, y) pairs

        gesture = hg.detect_gesture(points)
        print(gesture, "->", hg.gesture_to_action(gesture))
```

## Files in this folder

| File | What it holds |
|------|---------------|
| `hand_gestures.py` | the gesture math, the only file you really need |
| `sample_hands.py` | fake hand poses so everything runs without a camera |
| `demo.py` | a short script that prints the gesture for each sample pose |
| `test_hand_gestures.py` | small tests, run with plain Python or with pytest |

## Where this fits in the full app

In the full mHumanTrackAI app this same idea grows into a lot more:

* a Python engine that reads full body, both hands, and every finger from one or more webcams
* fast whole body tracking on the GPU, with a CPU fallback
* a guess at where a joint is when the camera cannot see it, so tracking does not jump
* a desktop **Accessibility Mode** for cursor, clicks, scroll, and typing helpers
* a **Gaming Mode** that acts as a virtual game controller, with an option for VR
* a friendly app where you watch yourself and the AI side by side and edit your own gestures

See it all at **https://mhumantrackai.m4herx.com**.

## License

This sample folder is released under the **MIT License**, so you are free to use it,
change it, and build on it. See [`LICENSE`](LICENSE).

The full mHumanTrackAI application is a separate product with its own license and is not
covered by the MIT license in this folder.
