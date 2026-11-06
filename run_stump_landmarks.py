#!/usr/bin/env python3
"""Stump landmarks (models/stump_landmarks_v01.onnx): top + base of each of the 6 stumps (12 points).

    python3 run_stump_landmarks.py                    # every image in images/
    python3 run_stump_landmarks.py images/ply2.jpg    # one image
    python3 run_stump_landmarks.py videos/v1.mp4      # a video -> annotated video
Output: output/stump_landmarks/
"""
from crickcat import core as C, runner as R


def summarize(pts, im):
    f = [p for p in pts if p["xy"] is not None]
    near = sum(p["end"] == "near" for p in f); far = len(f) - near
    return dict(headline=f"{len(f)}/12 landmarks (near wicket {near}/6, far wicket {far}/6)",
                confidence=C.conf_stats([p["conf"] for p in f]),
                all_12_conf=C.conf_stats([p["conf"] for p in pts]),
                missing=[f"{p['end']} {p['stump']} {p['part']}" for p in pts if p["xy"] is None])


if __name__ == "__main__":
    a = R.parse(__doc__, 0.25)
    R.run("stump_landmarks", "STUMP LANDMARKS", a, C.StumpLandmarks, lambda m, im, conf: m.detect(im, conf),
          C.draw_landmarks, summarize, C.REFERENCE_ACCURACY["stump_landmarks"])
