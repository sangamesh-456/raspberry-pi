#!/usr/bin/env python3
"""Ball detection + tracking, the Android app's default chain, plus the analyses built on the track.

    stage 1  models/stage1_btv7neg_ep29.onnx   finds the ball in every frame (288x512, 3 frames in / 3 out)
    stage 2  models/refiner_ens_a + _b         re-localises it in a 128x128 full-resolution crop
             coasting + FlightGate              fills short gaps, keeps only the delivery's flight
    3D       models/stump.tflite + arcfit.py    camera from the stumps -> speed, length, line
    batting  models/batting_contact / _slog     Hit / Miss, Lofted / Grounded

    python3 run_ball_tracking.py                       # every video in videos/
    python3 run_ball_tracking.py videos/v1.mp4         # one video
    python3 run_ball_tracking.py videos/v1.mp4 --release-ms 1200 --pitch-len 20.12

Output: output/ball_tracking/<video name>/
    tracked.mp4      the video with the ball trail drawn
    trajectory.jpg   one frame with the whole path
    report.json / report.txt   time, confidence, track quality, speed / length / line, batting
    points.csv       frame, x, y, confidence, how the point was found
"""
import argparse
import os
import re

import cv2
import numpy as np

from crickcat import core as C


def release_frame(args, path, fps):
    ms = args.release_ms
    if ms is None:
        m = re.search(r"_r(\d+)", os.path.basename(path))         # app clips are named ..._r<ms>.mp4
        ms = int(m.group(1)) if m else None
    return None if ms is None else int(ms / 1000 * fps)


def analyse(path, tracker, load_ms, args):
    tm = C.Timer()
    n, fps, w, h = C.video_meta(path)
    rel = release_frame(args, path, fps)
    start = max(0, rel - int(0.4 * fps)) if rel else 0             # app: track from release - 0.4 s
    t = C.now()
    gated, ungated, src, st, _ = tracker.track(path, gate=args.gate, from_frame=start, tm=tm)
    if not args.no_ensemble and st["kept"] < 8 and args.gate > 0.5:  # app: second pass at a looser gate
        tm.ms.clear()
        gated, ungated, src, st, _ = tracker.track(path, gate=0.5, from_frame=start, tm=tm)
        st["fallback_pass_gate_0.5"] = True
    track_ms = C.ms(t)
    frames = n - start
    timing = dict(model_load_ms=round(load_ms), track_total_s=round(track_ms / 1000, 2),
                  ms_per_frame=round(track_ms / max(frames, 1), 1),
                  tracking_fps=round(frames / (track_ms / 1000), 2),
                  video_duration_s=round(frames / fps, 2),
                  realtime_factor=round((frames / fps) / (track_ms / 1000), 2))
    pts = [(i, p) for i, p in enumerate(gated) if p is not None]
    confs = [p[2] for _, p in pts]
    by_src = {}
    for i, _ in pts:
        by_src.setdefault(src[i] or "detected", []).append(gated[i][2])
    rep = dict(input=os.path.basename(path), frames=n, fps=round(fps, 2), size=[w, h],
               release_frame=rel, tracked_from_frame=start, gate=args.gate,
               refiners="ensemble A+B" if not args.no_ensemble else "refiner_v01",
               timing=timing, tracker=st,
               confidence=dict(all_points=C.conf_stats(confs),
                               by_source={k: C.conf_stats(v) for k, v in by_src.items()},
                               ball_found_in_frames_pct=round(100 * len(pts) / max(frames, 1), 1)),
               track_2d=C.track_2d_stats(gated, fps))
    if not args.no_3d:
        rep["speed_length_line_3d"] = C.arc_3d(path, ungated, fps, w, rel, args.pitch_len, tm)
    rep["batting"] = C.batting_cards(gated, tm)
    rep["batting"]["reference_accuracy"] = dict(contact=C.REFERENCE_ACCURACY["batting_contact"],
                                                slog=C.REFERENCE_ACCURACY["batting_slog"])
    rep["accuracy"] = ("not measurable on this video - there are no ground-truth ball labels for it. "
                       "Use confidence, track completeness and the 3D fit error (fit_rms_px) as quality signals.")
    od = C.out_dir("ball_tracking", C.stem(path))
    with open(os.path.join(od, "points.csv"), "w") as f:
        f.write("frame,time_s,x,y,confidence,source\n")
        for i, p in pts:
            f.write(f"{i},{i / fps:.3f},{p[0]:.1f},{p[1]:.1f},{p[2]:.3f},{src[i] or 'detected'}\n")
    if not args.no_video:
        t = C.now(); rep["output_video"] = draw_video(path, gated, rep, os.path.join(od, "tracked.mp4")); tm.add("render_video", t)
    t = C.now(); rep["output_image"] = draw_trajectory(path, gated, rep, os.path.join(od, "trajectory.jpg")); tm.add("render_image", t)
    rep["timing"]["breakdown_ms"] = {k: round(v) for k, v in tm.ms.items()}
    rep["system"] = C.system_info()
    C.write_json(os.path.join(od, "report.json"), rep)
    C.write_text(os.path.join(od, "report.txt"), text(rep))
    return rep


def _metrics_lines(r):
    a = r.get("speed_length_line_3d") or {}
    L = []
    if a.get("solved"):
        L.append(f"speed {a['speed_kmh']} km/h" + ("" if a["plausible"] else " (implausible)"))
        if "length_zone" in a:
            L.append(f"length {a['length_zone']} ({a['length_m_from_batter_stumps']} m)")
        if "at_stumps_line_m" in a:
            L.append(f"line {a['at_stumps_line_m']:+.2f} m  hitting {'yes' if a['hitting_stumps'] else 'no'}")
    b = r["batting"]
    if b.get("contact"):
        L.append(f"contact {b['contact']}" + (f"  {b['slog']}" if b.get("slog") else ""))
    return L


def _color(c):
    return (40, 40, 230) if c >= 0.9 else (0, 165, 255) if c >= 0.75 else (0, 230, 255)


def draw_video(path, pts, r, out):
    n, fps, w, h = C.video_meta(path)
    cap = cv2.VideoCapture(path)
    vw, out = C.video_writer(out, fps, w, h)
    th = max(3, w // 300); trail, i = [], 0
    last = max([k for k, p in enumerate(pts) if p is not None], default=-1)
    while True:
        ok, im = cap.read()
        if not ok:
            break
        p = pts[i] if i < len(pts) else None
        if p is not None:
            trail.append((int(p[0]), int(p[1]), p[2]))
        for a, b in zip(trail, trail[1:]):
            cv2.line(im, a[:2], b[:2], _color(b[2]), th, cv2.LINE_AA)
        lines = [f"CrickCAT ball tracking  frame {i}/{n}"]
        if p is not None:
            cv2.circle(im, (int(p[0]), int(p[1])), th * 4, (255, 255, 255), max(2, th // 2), cv2.LINE_AA)
            lines.append(f"ball conf {p[2]:.2f}")
        else:
            lines.append("ball: -")
        if last >= 0 and i >= last:
            lines += _metrics_lines(r)
        C.hud(im, lines)
        vw.write(im); i += 1
    cap.release(); vw.release()
    return out


def draw_trajectory(path, pts, r, out):
    idx = [i for i, p in enumerate(pts) if p is not None]
    im = C.read_frame(path, idx[len(idx) // 2] if idx else 0)
    if im is None:
        return None
    w = im.shape[1]; th = max(3, w // 300)
    for a, b in zip(idx, idx[1:]):
        cv2.line(im, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), _color(pts[b][2]), th, cv2.LINE_AA)
    for i in idx:
        cv2.circle(im, (int(pts[i][0]), int(pts[i][1])), th, (255, 255, 255), -1, cv2.LINE_AA)
    if idx:
        C.label(im, f"first f{idx[0]}", (pts[idx[0]][0] + 10, pts[idx[0]][1]), (60, 220, 60))
        C.label(im, f"last f{idx[-1]}", (pts[idx[-1]][0] + 10, pts[idx[-1]][1]), (60, 220, 60))
        bf = r["track_2d"].get("image_bounce_frame")
        if bf is not None and pts[bf] is not None:
            C.label(im, "bounce", (pts[bf][0] + 10, pts[bf][1]), (255, 140, 40))
    c = r["confidence"]["all_points"]
    C.hud(im, [f"{r['input']}: {r['tracker']['kept']} ball points, conf mean {c['mean']}",
               f"tracked in {r['timing']['track_total_s']} s ({r['timing']['ms_per_frame']} ms/frame)"] + _metrics_lines(r))
    cv2.imwrite(out, im)
    return out


def text(r):
    t = r["timing"]; s = r["tracker"]; t2 = r["track_2d"]; c = r["confidence"]
    L = [f"=== BALL TRACKING: {r['input']} ({r['frames']} frames @ {r['fps']} fps, {r['size'][0]}x{r['size'][1]})",
         f"  release frame   : {r['release_frame']} (tracking from frame {r['tracked_from_frame']})",
         "  -- time --",
         f"  model load      : {t['model_load_ms']} ms",
         f"  detect + track  : {t['track_total_s']} s for {t['video_duration_s']} s of video "
         f"({t['ms_per_frame']} ms/frame, {t['tracking_fps']} fps, {t['realtime_factor']}x real time)",
         f"  breakdown (ms)  : {t['breakdown_ms']}",
         "  -- detection --",
         f"  stage-1 windows {s['windows']}, refined {s['refined']}, refused {s['dropped']}, "
         f"coasted {s['coasted']}/{s['coast_tried']}, removed by flight gate {s['rejected_by_flight_gate']}",
         f"  ball points kept: {s['kept']}  ({c['ball_found_in_frames_pct']}% of tracked frames)",
         "  -- confidence --",
         f"  all points      : mean {c['all_points']['mean']}  min {c['all_points']['min']}  max {c['all_points']['max']}"]
    for k, v in c["by_source"].items():
        L.append(f"  {k:16s}: n={v['n']}  mean {v['mean']}")
    if t2.get("points"):
        L += ["  -- 2D track --",
              f"  ball visible    : frames {t2['first_frame']}..{t2['last_frame']} ({t2['visible_s']} s), "
              f"completeness {t2['completeness']:.0%}",
              f"  speed in image  : median {t2['px_per_frame_median']} px/frame, image bounce frame {t2['image_bounce_frame']}",
              f"  start -> end    : {t2['start_xy']} -> {t2['end_xy']}"]
    a = r.get("speed_length_line_3d")
    if a is not None:
        L.append("  -- speed / length / line (3D) --")
        if a.get("solved"):
            L.append(f"  speed           : {a['speed_kmh']} km/h  plausible={a['plausible']}"
                     f"{'' if a['plausible'] else ' (' + str(a['reject_reason']) + ')'}")
            L.append(f"  fit error       : {a['fit_rms_px']} px rms over {a['points_used']} points")
            if "length_zone" in a:
                L.append(f"  length          : {a['length_zone']} - pitched {a['length_m_from_batter_stumps']} m from the batter's stumps")
            if "at_stumps_line_m" in a:
                L.append(f"  line at stumps  : {a['at_stumps_line_m']:+.2f} m ({a['line']}), height {a['at_stumps_height_m']:.2f} m")
                L.append(f"  hitting stumps  : {a['hitting_stumps']}")
        else:
            L.append(f"  {a.get('error')}")
    b = r["batting"]
    L += ["  -- batting --",
          f"  contact         : {b.get('contact')}  (confidence {b.get('contact_conf', '-')})" + (f"  {b['note']}" if b.get("note") else ""),
          f"  slog type       : {b.get('slog', '-')}  (confidence {b.get('slog_conf', '-')})",
          "  -- accuracy --",
          f"  this video      : {r['accuracy']}",
          f"  batting offline : contact {b['reference_accuracy']['contact']}; slog {b['reference_accuracy']['slog']}",
          "  -- files --",
          f"  {r.get('output_video')}", f"  {r.get('output_image')}",
          f"  device          : {r['system']['device']} ({r['system']['threads']} threads)"]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="a video or a folder (default: the videos/ folder)")
    ap.add_argument("--release-ms", type=int, help="ms from clip start to the ball's release (else _r<ms> in the name, else whole clip)")
    ap.add_argument("--gate", type=float, default=0.70, help="stage-1 confidence gate (app default 0.70)")
    ap.add_argument("--no-ensemble", action="store_true", help="one refiner instead of two (faster, finds the ball less often)")
    ap.add_argument("--no-3d", action="store_true", help="skip speed / length / line")
    ap.add_argument("--no-video", action="store_true", help="do not write tracked.mp4 (saves time)")
    ap.add_argument("--pitch-len", type=float, default=20.12, help="stump-to-stump distance in metres (scales speed)")
    ap.add_argument("--threads", type=int, default=C.THREADS)
    args = ap.parse_args()
    C.THREADS = args.threads
    vids = C.list_inputs(args.input, C.VIDEOS, C.VID_EXT)
    t = C.now(); tracker = C.TwoStage(ensemble=not args.no_ensemble); load_ms = C.ms(t)
    print(f"[ball_tracking] models loaded in {load_ms:.0f} ms\n")
    rows = []
    for v in vids:
        r = analyse(v, tracker, load_ms, args)
        a = r.get("speed_length_line_3d") or {}
        rows.append(dict(input=r["input"], track_s=r["timing"]["track_total_s"], ms_per_frame=r["timing"]["ms_per_frame"],
                         points=r["tracker"]["kept"], conf_mean=r["confidence"]["all_points"]["mean"],
                         speed_kmh=a.get("speed_kmh"), plausible=a.get("plausible"), length=a.get("length_zone"),
                         line_m=a.get("at_stumps_line_m"), contact=r["batting"].get("contact")))
        print()
    C.write_csv(os.path.join(C.out_dir("ball_tracking"), "summary.csv"), rows)
    print(f"[ball_tracking] done: {len(vids)} video(s). Summary: {os.path.join(C.out_dir('ball_tracking'), 'summary.csv')}")


if __name__ == "__main__":
    main()
