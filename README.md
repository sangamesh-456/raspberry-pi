# CrickCAT models on a Raspberry Pi

This folder holds everything needed to run every CrickCAT on-device model on a Raspberry Pi: the
models, the code, sample inputs and an output folder. The scripts use the same model files and the
same pre- and post-processing as the Android app.

```
raspberry-pi/
├── images/                  put test IMAGES here   (stump / player / landmark / pose detection)
├── videos/                  put test VIDEOS here   (ball tracking, clip action)
├── output/                  results appear here, one folder per model
├── models/                  the 11 model files (56 MB), copied from the app
├── crickcat/                shared code (do not run directly)
├── setup.sh                 one-time install
├── check_setup.py           checks packages + loads every model
├── run.sh                   runs a script with this folder's Python environment
├── run_ball_tracking.py     ball detection + tracking + speed / length / line + batting
├── run_clip_action.py       is the clip a delivery? (P(delivery))
├── run_stump_detection.py   stumps + bails boxes, pitch lock
├── run_player_detection.py  batter / bowler / wicket-keeper / fielder / umpire
├── run_stump_landmarks.py   12 stump points (top + base of each stump)
├── run_batter_pose.py       17 body keypoints per person
└── run_all.py               every model, one after the other
```

## 1. Copy to the Pi

Copy the whole `raspberry-pi` folder into `~/Downloads/` on the Pi (USB stick, or from a computer):

```bash
scp -r raspberry-pi admin@<pi-ip>:~/Downloads/
```

## 2. Install (once, needs internet)

```bash
cd ~/Downloads/raspberry-pi
bash setup.sh
```

This creates `venv/`, installs `numpy opencv-python-headless onnxruntime ai-edge-litert` (falling back
to `tflite-runtime`), and ends with `check_setup.py` printing **READY**.

## 3. Run each model

Add your files to `images/` and `videos/`, then:

```bash
cd ~/Downloads/raspberry-pi
./run.sh run_ball_tracking.py          # every video in videos/
./run.sh run_clip_action.py            # every video in videos/
./run.sh run_stump_detection.py        # every image in images/
./run.sh run_player_detection.py       # every image in images/
./run.sh run_stump_landmarks.py        # every image in images/
./run.sh run_batter_pose.py            # every image in images/
./run.sh run_all.py                    # all six, with a timing table
```

`./run.sh x.py` is the same as `./venv/bin/python x.py`. If the venv is activated
(`source venv/bin/activate`), `python3 x.py` works too.

You can also run one file. The detection scripts accept a video as well, and then write an annotated
video:

```bash
./run.sh run_ball_tracking.py videos/v1.mp4
./run.sh run_player_detection.py images/ply2.jpg
./run.sh run_player_detection.py videos/v2.mp4 --every 3       # model on every 3rd frame
./run.sh run_stump_landmarks.py videos/v1.mp4 --max-frames 30
```

### Options

| script | option | meaning |
|---|---|---|
| all | `--threads 4` | CPU threads (the Pi 5 has 4 cores) |
| detection | `--conf 0.25` | minimum confidence to keep a detection (pose default 0.15) |
| detection | `--keep-bars` | do not trim black letterbox bars before inference (trimmed by default) |
| detection (video) | `--every N`, `--max-frames N` | run on every Nth frame / stop after N frames |
| ball tracking | `--release-ms 1200` | ms from clip start to the ball's release. Tracking starts 0.4 s before it. Without it, the whole clip is tracked |
| ball tracking | `--pitch-len 20.12` | stump-to-stump distance at the venue. Every speed scales with it |
| ball tracking | `--no-ensemble` | one refiner instead of two: faster, finds the ball less often |
| ball tracking | `--no-3d` / `--no-video` | skip speed/length/line / skip writing `tracked.mp4` |
| clip action | `--no-video` | skip writing the annotated video |

## 4. What you get

**Ball tracking**, in `output/ball_tracking/<video>/`:

| file | contents |
|---|---|
| `tracked.mp4` | the video with the ball trail (red = high confidence, orange/yellow = lower) and a live panel; the analysis appears when the track ends |
| `trajectory.jpg` | one frame with the whole path, first/last frame and bounce marked |
| `report.txt` / `report.json` | the full analysis (below) |
| `points.csv` | frame, time, x, y, confidence, and how each point was found (detected / coasted / back-filled) |

`output/ball_tracking/summary.csv` has one row per video.

The report contains:
- **Time:** model load, total detect+track time, ms per frame, tracking fps, real-time factor, plus a
  breakdown (decode, stage 1, refiner, flight gate, pitch lock, 3D fit, batting, rendering).
- **Detection:** the number of stage-1 windows, points refined or refused, coasted gap fills, points
  the flight gate removed, and the share of frames where the ball was found.
- **Confidence:** mean, min and max over all ball points, and by how each point was found.
- **2D track:** visible frames and seconds, completeness, image speed in px/frame, the bounce frame.
- **Speed / length / line (3D):** speed in km/h, whether the fit is plausible, fit error in px rms,
  the length zone (yorker / full / good / short) with metres from the batter's stumps, the line and
  height at the stumps, and whether the ball was hitting the stumps. This needs stumps visible at
  **both** ends. Don't quote a speed marked `plausible=False`.
- **Batting:** Contact (Hit / Miss) and, on a Hit, Slog Type (Lofted / Grounded), each with a confidence.

**Detection scripts**, in `output/<model>/`: `<name>_result.jpg` (or `.mp4`) with boxes, points or
skeletons drawn, `<name>_report.txt` / `.json` and `summary.csv`. Each report has:
- **Time:** read, preprocess, inference, postprocess and draw/save in ms, the total, and model-only fps.
  Videos also get per-frame averages and pipeline fps.
- **Confidence:** the count, mean, min and max of what was found, plus per-class values (for example,
  stumps vs bails, or each player role).
- **Every detection:** its label, confidence and box or coordinates.

**Clip action**, in `output/clip_action/`: `P(delivery)` at its peak and the frame where it occurs,
the verdict (DELIVERY at ≥ 0.85, the app's green tick), the mean over the clip, a video with the
probability graph drawn live, and the peak frame as a jpg.

### About "accuracy"

You can only measure accuracy against hand-labelled ground truth, and your test images and videos
have none. So every report gives:
1. the model's **confidence** for this input, which is the per-run quality signal, and
2. the team's **offline accuracy**, where one was measured on labelled hold-out data:
   - clip action: precision 1.00 at recall 0.89
   - stump landmarks: median error 1.00 px, 90.6% of points within 4 px
   - batting contact: 0.758 accuracy (by venue)
   - batting slog type: 0.802 accuracy (by venue)

For ball tracking, the closest thing to an accuracy figure is the 3D fit error (`fit_rms_px`). It
says how well the tracked points fit a real ball flight under gravity.

## 5. Reference run (Windows laptop CPU, 4 threads, NOT a Pi)

`output/` already contains a run on the sample files. A Pi 5 will be slower. Running the scripts
overwrites these files, and you can delete them.

| model | input | result | time |
|---|---|---|---|
| ball tracking | v1.mp4 (93 fr, 1080x1920) | 45 points, conf 0.92, 75.0 km/h, yorker, hitting, Hit / Grounded | 3.98 s (42.8 ms/frame) |
| ball tracking | v2.mp4 (96 fr @ 24 fps) | 30 points, conf 0.96, 110.2 km/h, full, -0.50 m, Hit / Grounded | 3.69 s (38.5 ms/frame) |
| clip action | v1 / v2 | DELIVERY 0.970 / 0.927 | ~4.5 ms per window, ~0.8 s per clip |
| stump detection | ply1 / ply2 | 1 / 2 stump sets, no pitch lock | ~43 ms inference |
| player detection | ply1 / ply2 | batter 0.32 / 2 fielders | ~43 ms inference |
| stump landmarks | v1 frames | far wicket 6/6, near 0/6 | ~136 ms inference |
| batter pose | ply1 / ply2 | 2 / 1 people | ~10 ms inference |

## Things to know

- **The ball models work per delivery.** They expect a short clip (about 3–5 s) around one delivery.
  Cut a long recording into one clip per ball first.
- **Camera position matters.** The models were trained on phone video from behind the bowler's
  stumps, like `v1` and `v2`. Side-on views or a different lens find less.
- **Stump landmarks need the end-on view with both wickets.** On `ply1` and `ply2` they find nothing,
  and on `v1` they find the far wicket only (the near one is cut off at the bottom of the frame).
- **The player model labels poorly on these images.** It called the bowler a "fielder" on `ply2`.
- **The unit's own recordings** are in `/opt/crickcat-device/static/recordings`. Copy the clips you
  want into `videos/`.
