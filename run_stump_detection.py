#!/usr/bin/env python3
"""Stump detection (models/stump.tflite - YOLO, classes: stumps, bails) + the app's pitch-lock rule.

    python3 run_stump_detection.py                    # every image in images/
    python3 run_stump_detection.py images/ply2.jpg    # one image
    python3 run_stump_detection.py videos/v1.mp4      # a video -> annotated video
Output: output/stump_detection/
"""
from crickcat import core as C, runner as R


def summarize(dets, im):
    st = [d for d in dets if d["label"] == "stumps"]; bl = [d for d in dets if d["label"] == "bails"]
    lock = C.pitch_lock(dets, im.shape[0])
    return dict(headline=f"{len(st)} stump set(s), {len(bl)} bail(s), pitch lock {'YES' if lock else 'no'}",
                confidence=C.conf_stats([d["conf"] for d in dets]),
                stumps_conf=C.conf_stats([d["conf"] for d in st]), bails_conf=C.conf_stats([d["conf"] for d in bl]),
                pitch_lock=lock,
                pitch_lock_note="pitch lock = stumps found at BOTH ends (near + far); needed for speed / length / line")


def draw(im, dets):
    C.draw_boxes(im, dets)
    lock = C.pitch_lock(dets, im.shape[0])
    if lock:
        for k, c in (("near", (0, 220, 255)), ("far", (255, 140, 40))):
            x1, y1, x2, y2 = lock[k]
            C.label(im, k.upper(), (x1, y2 + 30), c)
    return im


if __name__ == "__main__":
    a = R.parse(__doc__, 0.25)
    R.run("stump_detection", "STUMP DETECTION", a, lambda: C.YoloTFLite("stump.tflite", C.STUMP_LABELS),
          lambda m, im, conf: m.detect(im, conf), draw, summarize)
