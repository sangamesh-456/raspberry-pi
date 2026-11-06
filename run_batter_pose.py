#!/usr/bin/env python3
"""Pose (models/batter_pose.onnx - YOLO-pose, 17 body keypoints per person).

    python3 run_batter_pose.py                        # every image in images/
    python3 run_batter_pose.py images/ply1.jpg        # one image
    python3 run_batter_pose.py videos/v1.mp4          # a video -> annotated video
Output: output/batter_pose/
"""
from crickcat import core as C, runner as R


def summarize(people, im):
    kp = [q[2] for p in people for q in p["keypoints"].values()]
    vis = [sum(q[2] >= 0.3 for q in p["keypoints"].values()) for p in people]
    return dict(headline=f"{len(people)} person(s), {vis} visible keypoints of 17",
                confidence=C.conf_stats([p["conf"] for p in people]),
                keypoint_conf=C.conf_stats(kp))


if __name__ == "__main__":
    a = R.parse(__doc__, 0.15)
    R.run("batter_pose", "BATTER POSE", a, C.BatterPose, lambda m, im, conf: m.detect(im, conf),
          C.draw_pose, summarize)
