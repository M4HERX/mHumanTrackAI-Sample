"""
Small test for the hand gesture reader.

These tests use the fake hands from sample_hands.py, so they need no camera and no
extra packages. You can run them two ways:

    pytest                  if you have pytest installed
    python test_hand_gestures.py   plain Python, prints a pass or fail summary

Every test checks one clear thing, so a failure tells you exactly what broke.
"""

from __future__ import annotations

import hand_gestures as hg
import sample_hands as hands


# which fingers are up

def test_open_palm_has_all_fingers_up():
    states = hg.fingers_up(hands.open_palm())
    assert hg.is_open_palm(states)
    assert not hg.is_fist(states)


def test_fist_has_all_fingers_down():
    states = hg.fingers_up(hands.fist())
    assert hg.is_fist(states)
    assert not hg.is_open_palm(states)


# pinch distance

def test_index_pinch_brings_thumb_and_index_close():
    points = hands.index_pinch()
    # the two tips are touching, so the gap is small
    assert hg.pinch_distance(points, "thumb", "index") < hg.PINCH_THRESHOLD


def test_open_palm_is_not_pinching():
    points = hands.open_palm()
    # fingers are spread, so no pair of tips is close enough to count as a pinch
    assert hg.pinch_distance(points, "thumb", "index") > hg.PINCH_THRESHOLD


def test_pinch_distance_is_the_same_near_or_far():
    """Move the whole hand closer to the camera and the pinch reading should not change,
    because we measure it as a fraction of the hand size."""
    near = hands.index_pinch()
    # shrink the hand toward the wrist to fake the hand being farther away
    wrist = near[0]
    far = [(wrist[0] + (x - wrist[0]) * 0.5, wrist[1] + (y - wrist[1]) * 0.5) for x, y in near]
    near_value = hg.pinch_distance(near, "thumb", "index")
    far_value = hg.pinch_distance(far, "thumb", "index")
    assert abs(near_value - far_value) < 0.05


# the one label per hand

def test_detect_gesture_labels_each_pose():
    assert hg.detect_gesture(hands.index_pinch()) == "IndexPinch"
    assert hg.detect_gesture(hands.middle_pinch()) == "MiddlePinch"
    assert hg.detect_gesture(hands.open_palm()) == "OpenPalm"
    assert hg.detect_gesture(hands.fist()) == "Fist"


def test_pinch_wins_over_open_palm():
    """An index pinch should read as a click even though the other fingers stay open."""
    assert hg.detect_gesture(hands.index_pinch()) == "IndexPinch"


# gesture to action

def test_index_pinch_maps_to_left_click():
    assert hg.gesture_to_action("IndexPinch") == "left click"


def test_unknown_gesture_does_nothing():
    assert hg.gesture_to_action("Wave") == "no action"


# A tiny runner so the file works even without pytest installed.
if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ok    {test.__name__}")
            passed += 1
        except AssertionError as error:
            print(f"  FAIL  {test.__name__}  {error}")
    print(f"\n{passed} of {len(tests)} tests passed")
