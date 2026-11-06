"""Shared loop for the image-detection scripts (stumps, players, landmarks, pose).

For every input it writes into output/<task>/:
    <name>_result.jpg   (or <name>_result.mp4 for a video)  - the picture with detections drawn
    <name>_report.json  - every detection + timing + confidence statistics
    <name>_report.txt   - the same, human-readable
and once per run:
    summary.csv         - one row per input
"""
import argparse
import os

import cv2

from . import core as C


def parse(desc, default_conf):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("input", nargs="?", help="an image, a video, or a folder (default: the images/ folder)")
    ap.add_argument("--conf", type=float, default=default_conf, help=f"confidence threshold (default {default_conf})")
    ap.add_argument("--threads", type=int, default=C.THREADS, help="CPU threads for inference (Pi 5 = 4)")
    ap.add_argument("--max-frames", type=int, default=0, help="video only: stop after N frames (0 = all)")
    ap.add_argument("--every", type=int, default=1, help="video only: run the model on every Nth frame")
    ap.add_argument("--keep-bars", action="store_true", help="do not trim black letterbox bars before inference")
    a = ap.parse_args()
    C.THREADS = a.threads
    return a


def _detect_trimmed(detect, model, im, args, box=None):
    """Run on the picture without its black letterbox bars, then shift results back."""
    x0, y0, x1, y1 = box or ((0, 0, im.shape[1], im.shape[0]) if args.keep_bars else C.content_box(im))
    items, tm = detect(model, im[y0:y1, x0:x1], args.conf)
    return C.shift(items, x0, y0), tm


def run(task, title, args, load, detect, draw, summarize, reference=None):
    """load() -> model; detect(model, im, conf) -> (items, timing); draw(im, items); summarize(items, im) -> dict."""
    inputs = C.list_inputs(args.input, C.IMAGES, C.IMG_EXT + C.VID_EXT)
    od = C.out_dir(task)
    t = C.now(); model = load(); load_ms = C.ms(t)
    print(f"[{task}] model loaded in {load_ms:.0f} ms  ->  output in {od}\n")
    rows = []
    for path in inputs:
        if path.lower().endswith(C.VID_EXT):
            rep = _video(path, od, model, args, detect, draw, summarize)
        else:
            rep = _image(path, od, model, args, detect, draw, summarize)
        rep["model_load_ms"] = round(load_ms, 1)
        rep["system"] = C.system_info()
        rep["accuracy"] = dict(
            this_input="not measurable - there is no ground-truth label for this input; use the confidence values",
            reference_offline=reference or "not recorded for this model")
        name = C.stem(path)
        C.write_json(os.path.join(od, f"{name}_report.json"), rep)
        C.write_text(os.path.join(od, f"{name}_report.txt"), _text(title, rep))
        print()
        rows.append(dict(input=rep["input"], type=rep["type"], **rep["summary_row"]))
    C.write_csv(os.path.join(od, "summary.csv"), rows)
    print(f"[{task}] done: {len(inputs)} input(s). Summary: {os.path.join(od, 'summary.csv')}")


def _image(path, od, model, args, detect, draw, summarize):
    t = C.now(); im = cv2.imread(path); read_ms = C.ms(t)
    if im is None:
        raise SystemExit(f"cannot read {path}")
    box = (0, 0, im.shape[1], im.shape[0]) if args.keep_bars else C.content_box(im)
    items, tm = _detect_trimmed(detect, model, im, args, box)
    t = C.now(); vis = draw(im.copy(), items)
    s = summarize(items, im)
    C.hud(vis, [f"{s['headline']}", f"inference {tm['inference_ms']:.0f} ms  total {read_ms + sum(tm.values()):.0f} ms",
                f"conf mean {s['confidence']['mean']}  max {s['confidence']['max']}"])
    out = os.path.join(od, f"{C.stem(path)}_result.jpg"); cv2.imwrite(out, vis); draw_ms = C.ms(t)
    timing = dict(read_ms=read_ms, **tm, draw_and_save_ms=draw_ms)
    timing["total_ms"] = sum(timing.values())
    timing["model_only_fps"] = 1000 / max(tm["inference_ms"], 1e-6)
    timing = {k: round(v, 1) for k, v in timing.items()}
    return dict(input=os.path.basename(path), type="image", size=[im.shape[1], im.shape[0]], output=out,
                model_input_region=list(box),
                conf_threshold=args.conf, timing=timing, summary=s, results=items,
                summary_row=dict(headline=s["headline"], conf_mean=s["confidence"]["mean"],
                                 conf_max=s["confidence"]["max"], inference_ms=timing["inference_ms"],
                                 total_ms=timing["total_ms"]))


def _video(path, od, model, args, detect, draw, summarize):
    n, fps, w, h = C.video_meta(path)
    cap = cv2.VideoCapture(path)
    out = os.path.join(od, f"{C.stem(path)}_result.mp4")
    vw, out = C.video_writer(out, fps, w, h)
    per, frames, last, i, box = [], [], [], 0, None
    agg = dict(read_ms=0.0, preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0, draw_and_save_ms=0.0)
    t_all = C.now()
    while True:
        t = C.now(); ok, im = cap.read(); agg["read_ms"] += C.ms(t)
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        if i % max(1, args.every) == 0:
            if box is None:
                box = (0, 0, w, h) if args.keep_bars else C.content_box(im)
            last, tm = _detect_trimmed(detect, model, im, args, box)
            for k, v in tm.items():
                agg[k] += v
            s = summarize(last, im)
            per.append(dict(frame=i, headline=s["headline"], conf_mean=s["confidence"]["mean"],
                            inference_ms=round(tm["inference_ms"], 1), results=last))
        t = C.now()
        vis = draw(im, last)
        C.hud(vis, [f"frame {i}/{n}", per[-1]["headline"] if per else "", f"inference {per[-1]['inference_ms'] if per else 0} ms"])
        vw.write(vis); agg["draw_and_save_ms"] += C.ms(t)
        i += 1
    cap.release(); vw.release()
    wall = C.ms(t_all)
    ran = max(len(per), 1)
    confs = [r.get("conf", 0) for p in per for r in p["results"] if "conf" in r and r.get("xy", 1) is not None]
    timing = dict(**{k: round(v, 1) for k, v in agg.items()}, total_ms=round(wall, 1),
                  frames_read=i, frames_inferred=len(per),
                  inference_ms_per_frame=round(agg["inference_ms"] / ran, 1),
                  pipeline_fps=round(i / (wall / 1000), 2) if wall else None,
                  model_only_fps=round(1000 / max(agg["inference_ms"] / ran, 1e-6), 2),
                  video_fps=round(fps, 2))
    found = [sum(1 for r in p["results"] if r.get("xy", 1) is not None) for p in per]
    headline = f"{sum(found) / ran:.1f} detections per frame on average (last frame: {per[-1]['headline'] if per else '-'})"
    s = dict(headline=headline, confidence=C.conf_stats(confs),
             frames_with_detections=sum(1 for f in found if f))
    return dict(input=os.path.basename(path), type="video", size=[w, h], output=out, conf_threshold=args.conf,
                timing=timing, summary=s, per_frame=per,
                summary_row=dict(headline=headline, conf_mean=s["confidence"]["mean"], conf_max=s["confidence"]["max"],
                                 inference_ms=timing["inference_ms_per_frame"], total_ms=timing["total_ms"]))


def _text(title, r):
    t = r["timing"]; s = r["summary"]
    L = [f"=== {title}: {r['input']} ({r['type']}, {r['size'][0]}x{r['size'][1]})",
         f"  result          : {s['headline']}",
         f"  output file     : {r['output']}",
         "  -- time --",
         f"  model load      : {r['model_load_ms']} ms (once per run)"]
    for k, v in t.items():
        L.append(f"  {k:16s}: {v}")
    c = s["confidence"]
    L += ["  -- confidence --",
          f"  detections      : {c['n']}   mean {c['mean']}   min {c['min']}   max {c['max']}   (threshold {r['conf_threshold']})"]
    for k, v in s.items():
        if k not in ("headline", "confidence"):
            L.append(f"  {k:16s}: {v}")
    if r["type"] == "image":
        L.append("  -- detections --")
        for d in r["results"]:
            L.append("  " + ", ".join(f"{k}={v}" for k, v in d.items() if k not in ("keypoints", "class_id")))
    L += ["  -- accuracy --",
          f"  this input      : {r['accuracy']['this_input']}",
          f"  offline (team)  : {r['accuracy']['reference_offline']}",
          f"  device          : {r['system']['device']}  ({r['system']['threads']} threads, onnxruntime {r['system']['onnxruntime']})"]
    return L
