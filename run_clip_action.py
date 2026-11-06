#!/usr/bin/env python3
"""Clip action (models/clip_action.onnx): is this clip a real delivery? P(delivery) through the clip.

    python3 run_clip_action.py                  # every video in videos/
    python3 run_clip_action.py videos/v2.mp4    # one video

Output: output/clip_action/
    <name>_result.mp4   the video with P(delivery) and a live graph drawn on it
    <name>_peak.jpg     the frame where the model is most sure
    <name>_report.json / _report.txt
"""
import argparse
import os

import cv2
import numpy as np

from crickcat import core as C

TICK = 0.85                      # app shows the green tick at >= 0.85


def verdict(p):
    return "DELIVERY" if p >= TICK else "unsure" if p >= 0.5 else "not a delivery"


def graph(im, curve, n, upto):
    h, w = im.shape[:2]; gh = h // 8; y0 = h - gh - 20
    ov = im.copy(); cv2.rectangle(ov, (10, y0 - 10), (w - 10, h - 10), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.6, im, 0.4, 0, im)
    ty = int(y0 + gh * (1 - TICK)); cv2.line(im, (20, ty), (w - 20, ty), (60, 220, 60), 1)
    pts = [(int(20 + (w - 40) * f / max(n - 1, 1)), int(y0 + gh * (1 - p))) for f, p in curve if f <= upto]
    for a, b in zip(pts, pts[1:]):
        cv2.line(im, a, b, (0, 220, 255), max(2, w // 400), cv2.LINE_AA)
    x = int(20 + (w - 40) * upto / max(n - 1, 1)); cv2.line(im, (x, y0), (x, y0 + gh), (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="a video or a folder (default: the videos/ folder)")
    ap.add_argument("--threads", type=int, default=C.THREADS)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(); C.THREADS = args.threads
    vids = C.list_inputs(args.input, C.VIDEOS, C.VID_EXT)
    od = C.out_dir("clip_action")
    t = C.now(); model = C.ClipAction(); load_ms = C.ms(t)
    rows = []
    for v in vids:
        tm = C.Timer()
        curve, nfr = model.score(v, tm)
        n, fps, w, h = C.video_meta(v)
        best_f, best = max(curve, key=lambda c: c[1]) if curve else (None, 0.0)
        ps = [p for _, p in curve]
        name = C.stem(v)
        rep = dict(input=os.path.basename(v), frames=nfr, fps=round(fps, 2), size=[w, h],
                   p_delivery=round(best, 3), verdict=verdict(best), peak_frame=best_f,
                   peak_time_s=round(best_f / fps, 2) if best_f is not None else None,
                   confidence=dict(peak=round(best, 3), mean_over_clip=round(float(np.mean(ps)), 3) if ps else None,
                                   frames_above_tick_pct=round(100 * sum(p >= TICK for p in ps) / max(len(ps), 1), 1)),
                   timing=dict(model_load_ms=round(load_ms), **{k + "_ms": round(v_) for k, v_ in tm.ms.items()},
                               windows_scored=len(curve),
                               ms_per_window=round(tm.ms.get("inference", 0) / max(len(curve), 1), 2)),
                   accuracy=dict(this_input="not measurable - no ground-truth label for this clip",
                                 reference_offline=C.REFERENCE_ACCURACY["clip_action"]),
                   curve=[[f, round(p, 4)] for f, p in curve], system=C.system_info())
        rep["timing"]["total_ms"] = round(sum(tm.ms.values()))
        if best_f is not None:
            im = C.read_frame(v, best_f)
            C.hud(im, [f"peak P(delivery) {best:.3f} at frame {best_f}", verdict(best)])
            cv2.imwrite(os.path.join(od, f"{name}_peak.jpg"), im)
        if not args.no_video:
            t = C.now()
            cap = cv2.VideoCapture(v); vw, out = C.video_writer(os.path.join(od, f"{name}_result.mp4"), fps, w, h)
            i, cur = 0, 0.0; cd = dict(curve)
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                cur = cd.get(i, cur)
                C.hud(im, [f"clip action  frame {i}/{n}", f"P(delivery) now {cur:.2f}", f"clip verdict: {verdict(best)} ({best:.2f})"])
                graph(im, curve, n, i); vw.write(im); i += 1
            cap.release(); vw.release(); rep["output_video"] = out; rep["timing"]["render_ms"] = round(C.ms(t))
        C.write_json(os.path.join(od, f"{name}_report.json"), rep)
        tt = rep["timing"]; c = rep["confidence"]
        C.write_text(os.path.join(od, f"{name}_report.txt"), [
            f"=== CLIP ACTION: {rep['input']} ({nfr} frames @ {rep['fps']} fps)",
            f"  verdict         : {rep['verdict']}   P(delivery) = {rep['p_delivery']} at frame {best_f} ({rep['peak_time_s']} s)",
            "  -- time --",
            f"  model load      : {tt['model_load_ms']} ms",
            f"  decode + resize : {tt.get('decode_and_resize_ms')} ms",
            f"  inference       : {tt.get('inference_ms')} ms for {tt['windows_scored']} windows ({tt['ms_per_window']} ms each)",
            f"  total           : {tt['total_ms']} ms",
            "  -- confidence --",
            f"  peak {c['peak']}   mean over clip {c['mean_over_clip']}   {c['frames_above_tick_pct']}% of windows >= {TICK}",
            "  -- accuracy --",
            f"  this input      : {rep['accuracy']['this_input']}",
            f"  offline (team)  : {rep['accuracy']['reference_offline']}",
            f"  output          : {rep.get('output_video')}"])
        print()
        rows.append(dict(input=rep["input"], verdict=rep["verdict"], p_delivery=rep["p_delivery"], peak_frame=best_f,
                         inference_ms=tt.get("inference_ms"), total_ms=tt["total_ms"]))
    C.write_csv(os.path.join(od, "summary.csv"), rows)
    print(f"[clip_action] done: {len(vids)} video(s). Summary: {os.path.join(od, 'summary.csv')}")


if __name__ == "__main__":
    main()
