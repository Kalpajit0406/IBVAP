"""
test_posture.py -- geometry checks for PoseClassifier, Rule 5 (weapon-ready
posture) in particular. No model, no GPU: synthetic 17x3 keypoint skeletons.

    python tests/test_posture.py        # standalone, prints PASS/FAIL, exits 1 on failure
    pytest tests/test_posture.py        # also works
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.posture import PoseClassifier  # noqa: E402

# COCO-17 indices
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14
L_AN, R_AN = 15, 16


def base_skeleton() -> np.ndarray:
    """A normal, upright, facing-camera person. Arms hang at the sides.

    Layout (image coords, y down): shoulders at y=90 span x 70..130
    (width 60), hips at y=190 -> body height 100.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    pts = {
        NOSE: (100, 40),
        L_SH: (70, 90),  R_SH: (130, 90),
        L_EL: (65, 130), R_EL: (135, 130),
        L_WR: (60, 175), R_WR: (140, 175),
        L_HIP: (80, 190), R_HIP: (120, 190),
        L_KN: (78, 260), R_KN: (122, 260),
        L_AN: (77, 330), R_AN: (123, 330),
    }
    for i, (x, y) in pts.items():
        kp[i] = (x, y, 1.0)
    return kp


def with_joints(**joints: tuple) -> np.ndarray:
    kp = base_skeleton()
    for name, (x, y) in joints.items():
        kp[globals()[name]] = (x, y, 1.0)
    return kp


def hold(clf: PoseClassifier, kp: np.ndarray, n: int, tid: int = 1):
    """Feed the same skeleton n times; return the final PostureFlags."""
    flags = None
    for _ in range(n):
        flags = clf.classify(tid, kp)
    return flags


# ── raw geometries (base: shoulders y90 span x70-130, hips y190, body_h 100) ──
AIM = with_joints(          # two-handed gun hold at chest, arms bent forward
    L_WR=(88, 95), R_WR=(112, 95),      # together, level, in band, centred
    L_EL=(72, 118), R_EL=(128, 118),    # elbows bent, up near torso height
)
AIM_LOW_READY = with_joints(  # gun held lower, muzzle down, elbows in
    L_WR=(90, 140), R_WR=(110, 145),
    L_EL=(76, 125), R_EL=(124, 128),
)
AIM_ANGLED = with_joints(   # aimed across-camera: hands shifted to one side
    L_WR=(120, 92), R_WR=(150, 96),
    L_EL=(95, 112), R_EL=(140, 116),
)
BELT = with_joints(         # hands clasped at the belt buckle — below the band
    L_WR=(94, 178), R_WR=(106, 178),
    L_EL=(74, 165), R_EL=(126, 165),
)
FOLDED = with_joints(       # arms folded: each wrist out past the far shoulder
    L_WR=(140, 120), R_WR=(60, 120),
    L_EL=(60, 118), R_EL=(140, 118),
)
SURRENDER = with_joints(L_WR=(70, 50), R_WR=(130, 50))     # both wrists above shoulders
ONE_HAND = with_joints(L_WR=(100, 95), R_WR=(140, 178))    # one forward, one at side
WIDE_HANDS = with_joints(L_WR=(40, 95), R_WR=(160, 95),    # chest height but far apart
                         L_EL=(45, 110), R_EL=(155, 110))
LYING = np.array([[10 + i * 12, 100, 1.0] for i in range(17)], dtype=np.float32)  # wide, flat


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("normal standing -> no aim")
def _(clf):
    f = hold(clf, base_skeleton(), 8)
    assert not f.chest_aim, f.label


@case("held two-handed aim fires after AIM_HOLD_FRAMES")
def _(clf):
    n = clf.AIM_HOLD_FRAMES
    early = [clf.classify(1, AIM) for _ in range(n - 1)]
    assert not any(e.chest_aim for e in early), "fired too early"
    assert clf.classify(1, AIM).chest_aim, "did not fire on the hold frame"


@case("low-ready gun hold (muzzle down) -> aim")
def _(clf):
    f = hold(clf, AIM_LOW_READY, clf.AIM_HOLD_FRAMES + 1)
    assert f.chest_aim


@case("gun aimed across-camera (hands to one side) -> aim")
def _(clf):
    f = hold(clf, AIM_ANGLED, clf.AIM_HOLD_FRAMES + 1)
    assert f.chest_aim


@case("aim streak resets on a non-aim frame")
def _(clf):
    hold(clf, AIM, clf.AIM_HOLD_FRAMES)          # fired
    clf.classify(1, base_skeleton())             # break the streak
    f = hold(clf, AIM, max(clf.AIM_HOLD_FRAMES - 1, 1))  # one short
    assert not f.chest_aim


@case("hands clasped at the belt (below the band) -> no aim")
def _(clf):
    f = hold(clf, BELT, 12)
    assert not f.chest_aim


@case("arms folded across chest -> no aim")
def _(clf):
    f = hold(clf, FOLDED, 12)
    assert not f.chest_aim


@case("surrender / hands-overhead -> arms_up, not aim")
def _(clf):
    f = hold(clf, SURRENDER, 12)
    assert f.arms_up and not f.chest_aim


@case("one hand forward, one at side -> no aim")
def _(clf):
    f = hold(clf, ONE_HAND, 12)
    assert not f.chest_aim


@case("hands at chest height but far apart -> no aim")
def _(clf):
    f = hold(clf, WIDE_HANDS, 12)
    assert not f.chest_aim


@case("flush_track clears the aim streak")
def _(clf):
    hold(clf, AIM, clf.AIM_HOLD_FRAMES - 1)
    clf.flush_track(1)
    f = hold(clf, AIM, clf.AIM_HOLD_FRAMES - 1)   # must start counting from zero again
    assert not f.chest_aim


@case("lying skeleton -> lying flag")
def _(clf):
    f = clf.classify(9, LYING)
    assert f.lying


def run() -> int:
    failed = 0
    for name, fn in CASES:
        clf = PoseClassifier()          # fresh state per case
        try:
            fn(clf)
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  -- {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


# pytest entry points
def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
