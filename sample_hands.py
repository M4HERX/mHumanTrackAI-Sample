"""
Fake hand data so you can try the code without a webcam.

Each function returns a list of 21 (x, y) points in the MediaPipe Hand order. The
numbers are made up by hand to look like real poses. Point 0 is the wrist near the
bottom, and fingers reach up toward smaller y values, the same way a real hand looks
when the camera is in front of you.

These are the exact poses the test file and the demo use.
"""

from __future__ import annotations


def _blank() -> list[list[float]]:
    """Start every point at the middle of the image, then we move the ones we care about."""
    return [[0.5, 0.5] for _ in range(21)]


def open_palm() -> list[tuple[float, float]]:
    """All five fingers stretched up and spread apart."""
    p = _blank()
    p[0] = [0.50, 0.90]  # wrist at the bottom

    # thumb reaches out to the side
    p[1] = [0.42, 0.82]
    p[2] = [0.36, 0.76]
    p[3] = [0.31, 0.70]
    p[4] = [0.26, 0.64]

    # the four fingers, each going straight up from its base
    fingers = {
        "index":  (0.44, 5),
        "middle": (0.50, 9),
        "ring":   (0.56, 13),
        "pinky":  (0.62, 17),
    }
    for x, base in fingers.values():
        p[base] = [x, 0.60]      # mcp, base knuckle
        p[base + 1] = [x, 0.45]  # pip, middle knuckle
        p[base + 2] = [x, 0.32]  # dip
        p[base + 3] = [x, 0.20]  # tip, well above the knuckles, so the finger reads as open

    return [tuple(point) for point in p]


def fist() -> list[tuple[float, float]]:
    """All four fingers curled down so each tip sits below its knuckle, near the wrist."""
    p = _blank()
    p[0] = [0.50, 0.90]
    p[9] = [0.50, 0.62]  # base of the middle finger, used for hand size

    # thumb wrapped across the front
    p[1] = [0.42, 0.80]
    p[2] = [0.40, 0.74]
    p[3] = [0.45, 0.72]
    p[4] = [0.49, 0.74]

    # for each finger the tip ends up lower (bigger y) than the middle knuckle, so it reads as curled
    fingers = {
        "index":  (0.45, 5),
        "middle": (0.50, 9),
        "ring":   (0.55, 13),
        "pinky":  (0.60, 17),
    }
    for x, base in fingers.values():
        p[base] = [x, 0.62]      # base knuckle
        p[base + 1] = [x, 0.64]  # middle knuckle
        p[base + 2] = [x, 0.70]  # dip, folding back down
        p[base + 3] = [x, 0.74]  # tip, tucked in close to the palm

    return [tuple(point) for point in p]


def _open_fingers(p: list[list[float]]) -> None:
    """Helper that opens the four fingers straight up, used by the pinch poses."""
    fingers = {
        "index":  (0.45, 5),
        "middle": (0.50, 9),
        "ring":   (0.55, 13),
        "pinky":  (0.60, 17),
    }
    for x, base in fingers.values():
        p[base] = [x, 0.60]
        p[base + 1] = [x, 0.48]
        p[base + 2] = [x, 0.38]
        p[base + 3] = [x, 0.30]


def index_pinch() -> list[tuple[float, float]]:
    """Thumb tip and index tip brought together. This is the left click pose."""
    p = _blank()
    p[0] = [0.50, 0.90]
    _open_fingers(p)

    # thumb reaches up toward the index tip
    p[1] = [0.43, 0.74]
    p[2] = [0.43, 0.62]
    p[3] = [0.44, 0.50]
    p[4] = [0.45, 0.31]  # thumb tip sits right next to the index tip at (0.45, 0.30)

    return [tuple(point) for point in p]


def middle_pinch() -> list[tuple[float, float]]:
    """Thumb tip and middle tip brought together. This is the right click pose."""
    p = _blank()
    p[0] = [0.50, 0.90]
    _open_fingers(p)

    # thumb reaches over to meet the middle fingertip at (0.50, 0.30)
    p[1] = [0.45, 0.74]
    p[2] = [0.46, 0.62]
    p[3] = [0.48, 0.50]
    p[4] = [0.50, 0.31]

    return [tuple(point) for point in p]


def point() -> list[tuple[float, float]]:
    """Index finger up on its own, the other fingers curled in. The move-the-cursor pose."""
    p = _blank()
    p[0] = [0.50, 0.90]  # wrist
    p[9] = [0.50, 0.60]  # base of the middle finger, used for hand size

    # thumb tucked in across the palm
    p[1] = [0.42, 0.80]
    p[2] = [0.40, 0.74]
    p[3] = [0.45, 0.72]
    p[4] = [0.49, 0.72]

    # index reaches straight up: its tip is well above its knuckles, so it reads as open
    p[5] = [0.45, 0.60]
    p[6] = [0.45, 0.48]
    p[7] = [0.45, 0.36]
    p[8] = [0.45, 0.24]

    # middle, ring and pinky curl down so each tip drops below its knuckle
    p[10] = [0.50, 0.62]
    p[11] = [0.50, 0.70]
    p[12] = [0.50, 0.74]
    p[13] = [0.55, 0.60]
    p[14] = [0.55, 0.62]
    p[15] = [0.55, 0.70]
    p[16] = [0.55, 0.74]
    p[17] = [0.60, 0.60]
    p[18] = [0.60, 0.62]
    p[19] = [0.60, 0.70]
    p[20] = [0.60, 0.74]

    return [tuple(point) for point in p]


def peace() -> list[tuple[float, float]]:
    """Index and middle up in a V, the other fingers curled in. The scroll pose."""
    p = _blank()
    p[0] = [0.50, 0.90]  # wrist
    p[9] = [0.50, 0.60]  # base of the middle finger, used for hand size

    # thumb tucked in across the palm
    p[1] = [0.42, 0.80]
    p[2] = [0.40, 0.74]
    p[3] = [0.45, 0.72]
    p[4] = [0.49, 0.72]

    # index and middle both reach straight up, so both read as open
    p[5] = [0.45, 0.60]
    p[6] = [0.45, 0.48]
    p[7] = [0.45, 0.36]
    p[8] = [0.45, 0.24]
    p[10] = [0.50, 0.48]
    p[11] = [0.50, 0.36]
    p[12] = [0.50, 0.24]

    # ring and pinky curl down so each tip drops below its knuckle
    p[13] = [0.55, 0.60]
    p[14] = [0.55, 0.62]
    p[15] = [0.55, 0.70]
    p[16] = [0.55, 0.74]
    p[17] = [0.60, 0.60]
    p[18] = [0.60, 0.62]
    p[19] = [0.60, 0.70]
    p[20] = [0.60, 0.74]

    return [tuple(point) for point in p]


# All the poses with their names, handy for looping in the demo.
ALL_POSES = {
    "open palm":   open_palm,
    "fist":        fist,
    "point":       point,
    "peace":       peace,
    "index pinch": index_pinch,
    "middle pinch": middle_pinch,
}
