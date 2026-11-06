#!/usr/bin/env python3
"""Run every model one after the other on the default folders and print one timing table.

    python3 run_all.py          # images/ for detection, videos/ for ball tracking + clip action
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [("ball tracking", "run_ball_tracking.py"), ("clip action", "run_clip_action.py"),
         ("stump detection", "run_stump_detection.py"), ("player detection", "run_player_detection.py"),
         ("stump landmarks", "run_stump_landmarks.py"), ("batter pose", "run_batter_pose.py")]
rows = []
for name, script in STEPS:
    print(f"\n{'#' * 80}\n# {name}\n{'#' * 80}")
    t = time.perf_counter()
    rc = subprocess.call([sys.executable, os.path.join(HERE, script)], cwd=HERE)
    rows.append((name, "ok" if rc == 0 else f"FAILED ({rc})", time.perf_counter() - t))
lines = ["", "=== ALL MODELS", f"  {'model':18s} {'status':12s} wall time (incl. loading + writing outputs)"]
lines += [f"  {n:18s} {s:12s} {w:6.1f} s" for n, s, w in rows]
lines.append(f"  outputs in {os.path.join(HERE, 'output')}  (each folder has summary.csv + per-input reports)")
print("\n".join(lines))
with open(os.path.join(HERE, "output", "run_all_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
