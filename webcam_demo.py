"""
Live webcam test for the gesture reader.

This opens your camera, tracks one hand with MediaPipe, and shows the gesture it reads
in real time. It is the same math as hand_gestures.py, just fed by a real camera instead
of fake poses. Hold up a hand and try a pinch, a fist, or an open palm.

Run it:
    python webcam_demo.py

Press Q or the Escape key to quit.

This file is what gets packaged into the downloadable .exe, so a user can test the idea
without setting up Python at all. The hand model file (hand_landmarker.task) sits next to
this script and is bundled into the .exe too.
"""

from __future__ import annotations

import os
import sys

import hand_gestures as hg

MODEL_FILE = "hand_landmarker.task"
WINDOW_NAME = "mHumanTrackAI gesture sample"

# The bones of a hand, as pairs of point numbers. We draw a line for each pair so the
# hand looks like a little skeleton on screen.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm base
]

# A colour for each gesture, in OpenCV's blue, green, red order.
GESTURE_COLOURS = {
    "IndexPinch":  (80, 220, 80),
    "MiddlePinch": (80, 200, 240),
    "OpenPalm":    (240, 200, 80),
    "Fist":        (80, 120, 240),
    "Hand":        (200, 200, 200),
}


def model_path() -> str:
    """Find the hand model whether we run from source or from inside the .exe.

    When PyInstaller bundles the app it unpacks files into a temp folder and points
    sys._MEIPASS at it, so we look there first. Running from source, the model just
    sits next to this script.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, MODEL_FILE)


def _fail(message: str) -> None:
    """Show a short message and wait, so a double clicked window does not just vanish."""
    print(message)
    try:
        input("\nPress Enter to close...")
    except EOFError:
        pass
    sys.exit(1)


def _make_tracker():
    """Build the MediaPipe hand tracker from the bundled model file."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    path = model_path()
    if not os.path.exists(path):
        _fail(f"Could not find the hand model next to the program:\n{path}")

    base = mp_python.BaseOptions(model_asset_path=path)
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def self_test() -> None:
    """Load everything and build the tracker without opening a camera.

    This is what the build uses to prove the packaged .exe is complete. If the model
    file or any library were left out of the bundle, this would fail, so a clean pass
    means the .exe will work on a real machine.
    """
    import cv2
    import mediapipe as mp

    tracker = _make_tracker()
    tracker.close()
    print(f"self test ok: cv2 {cv2.__version__}, mediapipe {mp.__version__}, hand model loaded")


def main() -> None:
    if "--check" in sys.argv:
        self_test()
        return

    # Import the heavy libraries inside main so a missing piece gives a clear message
    # instead of a wall of import errors.
    try:
        import cv2
        import mediapipe as mp
    except Exception as error:
        _fail(f"Could not load the camera libraries.\n{error}")

    tracker = _make_tracker()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        _fail("No camera found. Plug in a webcam and try again.")

    print("Camera is on. Hold up a hand. Press Q to quit.")
    timestamp = 0

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        # Mirror the frame so moving your hand right moves it right on screen.
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp += 33  # about 30 frames a second, in milliseconds
        result = tracker.detect_for_video(image, timestamp)

        gesture = "no hand"
        colour = (200, 200, 200)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            # Turn the MediaPipe landmarks into the plain (x, y) list our code expects.
            points = [(lm.x, lm.y) for lm in hand]

            gesture = hg.detect_gesture(points)
            colour = GESTURE_COLOURS.get(gesture, (200, 200, 200))

            _draw_hand(cv2, frame, points, colour)
            _draw_finger_states(cv2, frame, points)

        _draw_banner(cv2, frame, gesture, colour)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # Q or Escape
            break
        # Also stop when the window's X button is clicked. Without this, OpenCV just
        # reopens the window on the next frame and the demo seems impossible to close.
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    tracker.close()
    camera.release()
    cv2.destroyAllWindows()


def _draw_hand(cv2, frame, points, colour) -> None:
    """Draw the hand skeleton: a line for each bone and a dot at each point."""
    height, width = frame.shape[:2]
    pixels = [(int(x * width), int(y * height)) for x, y in points]

    for a, b in HAND_BONES:
        cv2.line(frame, pixels[a], pixels[b], colour, 2)
    for point in pixels:
        cv2.circle(frame, point, 4, (255, 255, 255), thickness=-1)


def _draw_banner(cv2, frame, gesture: str, colour) -> None:
    """Paint the gesture name and the action it would fire across the top of the frame."""
    action = hg.gesture_to_action(gesture) if gesture in hg.GESTURE_ACTIONS else ""
    height, width = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (width, 70), (30, 30, 30), thickness=-1)
    cv2.putText(frame, gesture, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
    if action:
        cv2.putText(frame, f"-> {action}", (320, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    cv2.putText(frame, "press Q to quit", (width - 230, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def _draw_finger_states(cv2, frame, points) -> None:
    """List each finger down the left side and mark it up or down."""
    states = hg.fingers_up(points)
    y = 110
    for finger in hg.FINGERS:
        up = states[finger]
        mark = "up  " if up else "down"
        colour = (80, 220, 80) if up else (120, 120, 120)
        cv2.putText(frame, f"{finger:7}{mark}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        y += 28


if __name__ == "__main__":
    main()
