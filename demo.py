"""
Tiny demo. Run it to see the gesture reader work on the fake hands.

    python demo.py

It walks through each sample pose, prints the 21 points it was given, the gesture it
found, and the action that gesture maps to. No camera needed.

To use this with a real webcam, install mediapipe, grab the 21 hand landmarks each
frame, turn them into a list of (x, y) pairs, and pass that to detect_gesture. The
README shows the few lines you need.
"""

from __future__ import annotations

import hand_gestures as hg
import sample_hands as hands


def main() -> None:
    print("mHumanTrackAI gesture sample\n")
    for name, make_hand in hands.ALL_POSES.items():
        points = make_hand()
        gesture = hg.detect_gesture(points)
        action = hg.gesture_to_action(gesture)

        print(f"pose given : {name}")
        print(f"fingers up : {hg.fingers_up(points)}")
        print(f"gesture    : {gesture}")
        print(f"action     : {action}")
        print()


if __name__ == "__main__":
    main()
