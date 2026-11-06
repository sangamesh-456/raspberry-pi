#!/usr/bin/env python3
"""Player detection (models/player.tflite - YOLO: batter, bowler, wicket-keeper, fielder, umpire).

    python3 run_player_detection.py                   # every image in images/
    python3 run_player_detection.py images/ply1.jpg   # one image
    python3 run_player_detection.py videos/v1.mp4     # a video -> annotated video
Output: output/player_detection/
"""
from crickcat import core as C, runner as R


def summarize(dets, im):
    counts = {l: sum(d["label"] == l for d in dets) for l in C.PLAYER_LABELS}
    found = ", ".join(f"{v} {k}" for k, v in counts.items() if v) or "nobody"
    return dict(headline=f"{len(dets)} player(s): {found}", confidence=C.conf_stats([d["conf"] for d in dets]),
                counts=counts,
                per_class_conf={l: C.conf_stats([d["conf"] for d in dets if d["label"] == l])["mean"]
                                for l in C.PLAYER_LABELS if counts[l]})


if __name__ == "__main__":
    a = R.parse(__doc__, 0.25)
    R.run("player_detection", "PLAYER DETECTION", a, lambda: C.YoloTFLite("player.tflite", C.PLAYER_LABELS),
          lambda m, im, conf: m.detect(im, conf), C.draw_boxes, summarize)
