"""
Read hand gestures from 21 hand landmarks.

This is the small math heart of mHumanTrackAI, pulled out so you can read it on its own.
A webcam tracker (like MediaPipe Hands) gives you 21 points for each hand. This file
turns those points into simple answers: which fingers are up, is the hand a fist or an
open palm, and are two fingers pinched together. From there you can map a pinch to a
mouse click or anything else you want.

There is no camera and no machine learning model in this file. It is just geometry, so
it runs anywhere and is easy to test.

The 21 points follow the MediaPipe Hand order. Each point is an (x, y) pair where x and
y go from 0 to 1 across the image. Point 0 is the wrist. The fingertips are points
4, 8, 12, 16 and 20.

    0  wrist
    1..4   thumb (tip is 4)
    5..8   index (tip is 8)
    9..12  middle (tip is 12)
    13..16 ring (tip is 16)
    17..20 pinky (tip is 20)
"""

from __future__ import annotations

import math

# The tip point of every finger.
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

# The middle knuckle of each finger. We compare the tip against this knuckle to tell
# if the finger is open or curled. The thumb bends a different way, so it has its own.
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
THUMB_IP = 3

WRIST = 0
MIDDLE_MCP = 9  # base of the middle finger, used to measure how big the hand looks
PINKY_MCP = 17  # base of the pinky, the far corner of the palm; the thumb's anchor

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Plain straight line distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_size(points: list[tuple[float, float]]) -> float:
    """
    Roughly how big the hand looks in the image, measured from the wrist to the base
    of the middle finger. We divide other distances by this number so the gestures
    work the same whether you sit close to the camera or far away. The tiny floor of
    1e-6 just keeps us from dividing by zero.
    """
    return max(1e-6, distance(points[WRIST], points[MIDDLE_MCP]))


def finger_is_open(points: list[tuple[float, float]], finger: str) -> bool:
    """
    True when the finger is stretched out, False when it is curled in.

    The trick: when a finger is open, its tip sits far from the wrist. When the finger
    curls, the tip drops back toward the wrist and ends up closer than the middle
    knuckle. So we just compare the two distances. This works even if the hand is
    tilted or rotated, because both distances rotate together.

    The thumb is the exception. It does not curl back toward the wrist like the other
    four fingers; it folds sideways across the palm, so its distance from the wrist
    barely changes and the wrist test cannot tell open from closed. Instead we anchor
    the thumb on the far corner of the palm, the base of the pinky. When the thumb is
    out, its tip reaches farther from that corner than its own knuckle does; when the
    thumb tucks in across the palm, the tip falls back inside the knuckle.

    The small 1.05 factor asks the tip to be a clear bit past the knuckle, so a half
    bent finger does not flicker between open and closed.
    """
    if finger == "thumb":
        tip_far = distance(points[PINKY_MCP], points[FINGER_TIPS["thumb"]])
        knuckle_far = distance(points[PINKY_MCP], points[THUMB_IP])
        return tip_far > knuckle_far * 1.05

    tip_far = distance(points[WRIST], points[FINGER_TIPS[finger]])
    knuckle_far = distance(points[WRIST], points[PIP[finger]])
    return tip_far > knuckle_far * 1.05


def fingers_up(points: list[tuple[float, float]]) -> dict[str, bool]:
    """Check every finger and return a simple map like {"index": True, ...}."""
    return {finger: finger_is_open(points, finger) for finger in FINGERS}


def is_fist(states: dict[str, bool]) -> bool:
    """A fist is when all four main fingers are curled. The thumb is allowed to stick out."""
    return not any(states[f] for f in ("index", "middle", "ring", "pinky"))


def is_open_palm(states: dict[str, bool]) -> bool:
    """An open palm is when every finger, thumb included, is stretched out."""
    return all(states[f] for f in FINGERS)


def pinch_distance(points: list[tuple[float, float]], finger_a: str, finger_b: str) -> float:
    """
    How close two fingertips are, as a fraction of the hand size. A value near 0 means
    the two tips are touching. A larger value means they are spread apart. Because we
    divide by the hand size, the number means the same thing near or far from the camera.
    """
    tip_a = points[FINGER_TIPS[finger_a]]
    tip_b = points[FINGER_TIPS[finger_b]]
    return distance(tip_a, tip_b) / hand_size(points)


# A pinch counts when the two tips are this close or closer. You can raise it to make
# pinching easier, or lower it to make it stricter.
PINCH_THRESHOLD = 0.35


def detect_gesture(points: list[tuple[float, float]]) -> str:
    """
    Turn a hand into one short label.

    We look for a pinch first, because a pinch is the click action and we want it to
    win even while the other fingers stay open. A tip touching the thumb only counts
    as a pinch when that finger is actually pointing out. This is what keeps a fist
    from looking like a pinch, since in a fist every finger is curled and its bunched
    tips do not count.

    If both the index and the middle look pinched at once, we pick the one whose tip
    the thumb is truly closest to. With no pinch we read which fingers are up: an
    open palm, a fist, a single pointing index, or a two finger peace sign, and fall
    back to a plain "Hand" when it is none of those.

    Returns one of:
        "IndexPinch"   thumb and index touching, used for a left click
        "MiddlePinch"  thumb and middle touching, used for a right click
        "Point"        index finger up on its own, used to move the cursor
        "Peace"        index and middle up together, used to scroll
        "OpenPalm"     all fingers spread, a good neutral or stop pose
        "Fist"         all fingers curled
        "Hand"         a hand is there but none of the above
    """
    states = fingers_up(points)

    index_gap = pinch_distance(points, "thumb", "index")
    middle_gap = pinch_distance(points, "thumb", "middle")
    index_pinching = states["index"] and index_gap < PINCH_THRESHOLD
    middle_pinching = states["middle"] and middle_gap < PINCH_THRESHOLD

    if index_pinching and middle_pinching:
        return "IndexPinch" if index_gap <= middle_gap else "MiddlePinch"
    if index_pinching:
        return "IndexPinch"
    if middle_pinching:
        return "MiddlePinch"

    if is_open_palm(states):
        return "OpenPalm"
    if is_fist(states):
        return "Fist"

    # index and middle up together, the other fingers curled: a peace sign, used to scroll
    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        return "Peace"
    # index up on its own, the other fingers curled: a pointing finger, moves the cursor
    if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
        return "Point"

    return "Hand"


# A friendly mapping from a gesture to what it would do on the desktop. The full app
# lets you edit these. Here we keep one simple default so the demo has something to show.
GESTURE_ACTIONS = {
    "IndexPinch": "left click",
    "MiddlePinch": "right click",
    "Point": "move cursor",
    "Peace": "scroll",
    "OpenPalm": "neutral (no action)",
    "Fist": "grab / hold",
    "Hand": "no action",
}


def gesture_to_action(gesture: str) -> str:
    """Look up what a gesture should do. Unknown gestures do nothing."""
    return GESTURE_ACTIONS.get(gesture, "no action")
