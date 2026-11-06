"""CrickCAT models on a Raspberry Pi: shared engine for every run_*.py script.

A Python port of what the Android app runs, using the SAME model files (copied into ../models)
and the same pre/post-processing (TwoStageTracker.kt, PatchRefiner.kt, FlightGate.kt, BattingCards.kt,
StumpLandmarks.kt, BatterPose.kt, ClipActionConfirmer.kt), so a number here means what it means on
the phone.
"""
import json
import math
import os
import platform
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
MODELS = os.environ.get("CRICKCAT_MODELS", os.path.join(ROOT, "models"))
IMAGES = os.path.join(ROOT, "images")
VIDEOS = os.path.join(ROOT, "videos")
OUTPUT = os.path.join(ROOT, "output")
sys.path.insert(0, PKG)                                   # arcfit.py, pitchcam.py
os.environ.setdefault("STUMP_TFLITE", os.path.join(MODELS, "stump.tflite"))

THREADS = min(4, os.cpu_count() or 4)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".h264", ".m4v")

PLAYER_LABELS = ["batter", "bowler", "wicket-keeper", "fielder", "umpire"]
STUMP_LABELS = ["stumps", "bails"]
KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
      "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
COLORS = [(0, 200, 255), (60, 220, 60), (255, 140, 40), (230, 60, 200), (40, 40, 230), (200, 200, 0)]

# Offline accuracy measured by the team on labelled hold-out data. This is NOT measured on your
# image: there is no ground truth for it. Per-run quality is the confidence the model reports.
REFERENCE_ACCURACY = {
    "clip_action": "precision 1.00 at recall 0.89 on held-out labelled sessions",
    "stump_landmarks": "median error 1.00 px, 90.6% of landmarks within 4 px (hold-out split by clip)",
    "batting_contact": "0.758 accuracy, leave-one-venue-out over 7 venues",
    "batting_slog": "0.802 accuracy, leave-one-venue-out over 3 venues",
}


# ------------------------------------------------------------------------------------------------
# plumbing
# ------------------------------------------------------------------------------------------------
def model_path(name):
    p = os.path.join(MODELS, name)
    if not os.path.exists(p):
        sys.exit(f"missing model {p}")
    return p


def onnx(name):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = THREADS
    o.log_severity_level = 3
    return ort.InferenceSession(model_path(name), o, providers=["CPUExecutionProvider"])


def tflite(name):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            sys.exit("TFLite runtime missing: pip install ai-edge-litert   (or: pip install tflite-runtime)")
    it = Interpreter(model_path=model_path(name), num_threads=THREADS)
    it.allocate_tensors()
    return it


def rnd(v):
    """Kotlin roundToInt (half up), not Python's banker's rounding."""
    return int(math.floor(v + 0.5))


def now():
    return time.perf_counter()


def ms(t0):
    return (time.perf_counter() - t0) * 1000


class Timer:
    def __init__(self):
        self.ms = {}

    def add(self, k, t0):
        self.ms[k] = self.ms.get(k, 0.0) + ms(t0)


def system_info():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/device-tree/model") as f:
            cpu = f.read().strip("\x00\n ")
    except OSError:
        pass
    return dict(device=cpu, machine=platform.machine(), os=platform.platform(), python=platform.python_version(),
                onnxruntime=ort.__version__, opencv=cv2.__version__, threads=THREADS)


def conf_stats(confs):
    if not confs:
        return dict(n=0, mean=None, min=None, max=None)
    a = np.array(confs, float)
    return dict(n=len(a), mean=round(float(a.mean()), 3), min=round(float(a.min()), 3), max=round(float(a.max()), 3))


def list_inputs(arg, default_dir, exts):
    """A file, a folder, or (default) the matching folder next to the scripts."""
    p = arg or default_dir
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        sys.exit(f"no such file or folder: {p}")
    fs = sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(exts))
    if not fs:
        sys.exit(f"no {'/'.join(exts)} files in {p}")
    return fs


def out_dir(task, name=None):
    d = os.path.join(OUTPUT, task, name) if name else os.path.join(OUTPUT, task)
    os.makedirs(d, exist_ok=True)
    return d


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def write_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)).replace(",", ";") for k in keys) + "\n")


def video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def read_frame(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, im = cap.read()
    cap.release()
    return im if ok else None


def video_writer(path, fps, w, h):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():                                  # some OpenCV builds lack mp4v
        path = os.path.splitext(path)[0] + ".avi"
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    return vw, path


def label(im, text, org, color, scale=None):
    scale = scale or max(0.5, im.shape[1] / 1400)
    th = max(1, int(scale * 2))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(max(tht + 4, org[1]))
    cv2.rectangle(im, (x, y - tht - 6), (x + tw + 6, y + 2), color, -1)
    cv2.putText(im, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_AA)


def hud(im, lines):
    """Translucent panel of text lines in the top-left corner."""
    s = max(0.5, im.shape[1] / 1500); th = max(1, int(s * 2)); lh = int(32 * s) + 6
    w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, s, th)[0][0] for l in lines) + 20
    ov = im.copy()
    cv2.rectangle(ov, (8, 8), (8 + w, 14 + lh * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.6, im, 0.4, 0, im)
    for i, l in enumerate(lines):
        cv2.putText(im, l, (18, 8 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), th, cv2.LINE_AA)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


# ------------------------------------------------------------------------------------------------
# DETECTORS (each returns (result, timing_ms) where timing = preprocess / inference / postprocess)
# ------------------------------------------------------------------------------------------------
class YoloTFLite:
    """stump.tflite / player.tflite: letterbox 640, raw YOLO head [4+nc, 8400] + per-class NMS."""

    def __init__(self, name, labels):
        self.it = tflite(name); self.labels = labels
        self.inp = self.it.get_input_details()[0]; self.out = self.it.get_output_details()[0]
        self.S = int(self.inp["shape"][2])

    def detect(self, im, conf=0.25, iou=0.45):
        t = now(); S = self.S; h, w = im.shape[:2]
        s = min(S / w, S / h); nw, nh = int(w * s), int(h * s)
        canvas = np.full((S, S, 3), 114, np.uint8); px, py = (S - nw) // 2, (S - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (nw, nh))
        x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        tp = ms(t); t = now()
        self.it.set_tensor(self.inp["index"], x); self.it.invoke()
        r = self.it.get_tensor(self.out["index"])[0]
        ti = ms(t); t = now()
        dets = []
        if r.shape[-1] == 6:                                    # end-to-end export [300, 6]
            for x1, y1, x2, y2, c, k in r:
                if c >= conf:
                    if x2 <= 1.5:
                        x1, y1, x2, y2 = x1 * S, y1 * S, x2 * S, y2 * S
                    dets.append([x1, y1, x2, y2, float(c), int(k)])
        else:                                                   # raw [4+nc, anchors]
            b = r[:4].T
            if np.abs(b).max() <= 1.5:
                b = b * S
            sc = r[4:].max(0); cl = r[4:].argmax(0); m = sc >= conf
            for (cx, cy, bw, bh), c, k in zip(b[m], sc[m], cl[m]):
                dets.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(c), int(k)])
        dets.sort(key=lambda d: -d[4]); keep = []
        for d in dets:
            if all(d[5] != k[5] or _iou(d, k) < iou for k in keep):
                keep.append(d)
        res = []
        for d in keep:
            bx = [(d[0] - px) / s, (d[1] - py) / s, (d[2] - px) / s, (d[3] - py) / s]
            bx = [round(min(max(bx[0], 0), w)), round(min(max(bx[1], 0), h)),
                  round(min(max(bx[2], 0), w)), round(min(max(bx[3], 0), h))]
            res.append(dict(label=self.labels[d[5]] if d[5] < len(self.labels) else str(d[5]),
                            class_id=d[5], conf=round(d[4], 3), box=bx))
        return res, dict(preprocess_ms=tp, inference_ms=ti, postprocess_ms=ms(t))


def draw_boxes(im, dets):
    th = max(2, im.shape[1] // 400)
    for d in dets:
        c = COLORS[d.get("class_id", 0) % len(COLORS)]
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(im, (x1, y1), (x2, y2), c, th)
        label(im, f"{d['label']} {d['conf']:.2f}", (x1, y1), c)
    return im


def pitch_lock(dets, H):
    """App rule: two stump boxes separated vertically -> near (lower) and far (upper) wicket."""
    st = [d for d in dets if d["label"] == "stumps"]
    if len(st) < 2:
        return None
    cy = [(d["box"][1] + d["box"][3]) / 2 / H for d in st]
    if max(cy) - min(cy) < 0.06:
        return None
    return dict(near=st[int(np.argmax(cy))]["box"], far=st[int(np.argmin(cy))]["box"])


class StumpLandmarks:
    """stump_landmarks_v01.onnx: 12 heatmaps = (near?6:0) + stump*2 + (top?1:0)."""

    def __init__(self):
        self.s = onnx("stump_landmarks_v01.onnx"); i = self.s.get_inputs()[0]
        self.name, self.H, self.W = i.name, i.shape[2], i.shape[3]

    def detect(self, im, conf=0.25):
        t = now(); H, W = im.shape[:2]
        x = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (self.W, self.H)).transpose(2, 0, 1)[None].astype(np.float32) / 255
        tp = ms(t); t = now()
        hm = self.s.run(None, {self.name: x})[0][0]
        ti = ms(t); t = now(); pts = []
        for c in range(12):
            j = int(np.argmax(hm[c])); y, xx = divmod(j, hm.shape[2]); v = float(hm[c].flat[j])
            pts.append(dict(end="near" if c >= 6 else "far", stump=["left", "middle", "right"][(c % 6) // 2],
                            part="top" if c % 2 else "base", conf=round(v, 3),
                            xy=[round((xx + 0.5) / hm.shape[2] * W), round((y + 0.5) / hm.shape[1] * H)] if v >= conf else None))
        return pts, dict(preprocess_ms=tp, inference_ms=ti, postprocess_ms=ms(t))


def draw_landmarks(im, pts):
    r = max(4, im.shape[1] // 160)
    by = {}
    for p in pts:
        if p["xy"] is None:
            continue
        c = (0, 220, 255) if p["end"] == "near" else (255, 140, 40)
        cv2.circle(im, tuple(p["xy"]), r, c, -1, cv2.LINE_AA)
        cv2.circle(im, tuple(p["xy"]), r, (0, 0, 0), 1, cv2.LINE_AA)
        by.setdefault((p["end"], p["stump"]), {})[p["part"]] = (p["xy"], c)
    for (end, _), d in by.items():
        if "top" in d and "base" in d:
            cv2.line(im, tuple(d["top"][0]), tuple(d["base"][0]), d["top"][1], max(1, r // 2), cv2.LINE_AA)
    for end in ("near", "far"):
        f = [p for p in pts if p["end"] == end and p["xy"]]
        if f:
            x = min(p["xy"][0] for p in f); y = min(p["xy"][1] for p in f)
            label(im, f"{end} wicket {len(f)}/6", (x, y - 2 * r), (0, 220, 255) if end == "near" else (255, 140, 40))
    return im


class BatterPose:
    """batter_pose.onnx (YOLO-pose 320, top-left letterbox like BatterPose.kt), conf 0.15, NMS 0.45."""
    S = 320

    def __init__(self):
        self.s = onnx("batter_pose.onnx"); self.name = self.s.get_inputs()[0].name

    def detect(self, im, conf=0.15, iou=0.45):
        t = now(); S = self.S
        h, w = im.shape[:2]; sc = min(S / h, S / w); nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (nw, nh))
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
        tp = ms(t); t = now()
        o = self.s.run(None, {self.name: x})[0][0]
        ti = ms(t); t = now()
        idx = sorted(np.where(o[4] >= conf)[0], key=lambda i: -o[4][i]); keep = []

        def box(k):
            return [o[0][k] - o[2][k] / 2, o[1][k] - o[3][k] / 2, o[0][k] + o[2][k] / 2, o[1][k] + o[3][k] / 2]
        for i in idx:
            if all(_iou(box(i), box(j)) <= iou for j in keep):
                keep.append(i)
        people = [dict(conf=round(float(o[4][i]), 3), box=[round(v / sc) for v in box(i)],
                       keypoints={KP[k]: [round(float(o[5 + 3 * k][i] / sc)), round(float(o[6 + 3 * k][i] / sc)),
                                          round(float(o[7 + 3 * k][i]), 2)] for k in range(17)}) for i in keep]
        return people, dict(preprocess_ms=tp, inference_ms=ti, postprocess_ms=ms(t))


def draw_pose(im, people, kp_conf=0.3):
    th = max(2, im.shape[1] // 450); r = max(3, im.shape[1] // 250)
    for n, p in enumerate(people):
        c = COLORS[n % len(COLORS)]
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(im, (x1, y1), (x2, y2), c, th)
        label(im, f"person {p['conf']:.2f}", (x1, y1), c)
        k = [p["keypoints"][n_] for n_ in KP]
        for a, b in SKELETON:
            if k[a][2] >= kp_conf and k[b][2] >= kp_conf:
                cv2.line(im, tuple(k[a][:2]), tuple(k[b][:2]), c, th, cv2.LINE_AA)
        for q in k:
            if q[2] >= kp_conf:
                cv2.circle(im, tuple(q[:2]), r, (255, 255, 255), -1, cv2.LINE_AA)
    return im


# ------------------------------------------------------------------------------------------------
# BALL: stage 1 + refiner ensemble + coasting + FlightGate (TwoStageTracker.kt)
# ------------------------------------------------------------------------------------------------
class Refiner:
    """PatchRefiner.kt: 3 source-resolution 128x128 crops -> heatmap (+ partner averaged) -> blob centroid."""
    PATCH = 128

    def __init__(self, a, b=None):
        self.s = onnx(a)
        self.p = onnx(b) if b else None

    def refine(self, frames, cx, cy, gate):
        P = self.PATCH
        h0, w0 = frames[0].shape[:2]
        if w0 < P or h0 < P:
            return None
        x0 = min(max(rnd(cx - P / 2), 0), max(0, w0 - P))
        y0 = min(max(rnd(cy - P / 2), 0), max(0, h0 - P))
        x = np.concatenate([f[y0:y0 + P, x0:x0 + P].transpose(2, 0, 1) for f in frames], 0)
        x = (x.astype(np.float32) / 255.0)[None]
        outs = self.s.run(None, {self.s.get_inputs()[0].name: x})
        heat = outs[0].reshape(P, P).copy()
        off = outs[1].reshape(2, P, P) if len(outs) > 1 else None
        if self.p is not None:
            heat = 0.5 * (heat + self.p.run(None, {self.p.get_inputs()[0].name: x})[0].reshape(P, P))
        c = self._decode(heat, gate)
        if c is None:
            return None
        fx, fy = c
        if off is not None:
            iy = min(max(rnd(fy), 0), P - 1); ix = min(max(rnd(fx), 0), P - 1)
            fx += float(off[0, iy, ix]); fy += float(off[1, iy, ix])
        return x0 + fx, y0 + fy, float(heat.max())

    @staticmethod
    def _decode(buf, gate):
        P = buf.shape[0]
        pi = int(np.argmax(buf)); peak = float(buf.flat[pi])
        if peak < gate:
            return None
        thr = gate * 0.7
        seen = np.zeros(buf.shape, bool)
        y, x = divmod(pi, P)
        stack = [(y, x)]; seen[y, x] = True
        sx = sy = sw = 0.0
        while stack:
            y, x = stack.pop()
            w = float(buf[y, x]); sx += x * w; sy += y * w; sw += w
            for ny, nx in ((y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)):
                if 0 <= ny < P and 0 <= nx < P and not seen[ny, nx] and buf[ny, nx] > thr:
                    seen[ny, nx] = True; stack.append((ny, nx))
        return (sx / sw, sy / sw) if sw > 0 else None


class TwoStage:
    def __init__(self, ensemble=True):
        self.s1 = onnx("stage1_btv7neg_ep29.onnx")
        i = self.s1.get_inputs()[0]
        self.inName = i.name
        self.nIn = i.shape[1] // 3
        self.H, self.W = i.shape[2], i.shape[3]
        self.nOut = min(self.s1.get_outputs()[0].shape[1], self.nIn)
        self.ensemble = ensemble
        self.ref = (Refiner("refiner_ens_a_ludocf.onnx", "refiner_ens_b_unionns.onnx") if ensemble
                    else Refiner("refiner_v01.onnx"))

    def track(self, clip, gate=0.70, refine_gate=0.9, from_frame=0, tm=None):
        tm = tm or Timer()
        ens = self.ensemble
        coast, coast_gate = 15, (0.98 if ens else 0.96)
        bridge = back = ens
        n_total, fps, sw, sh = video_meta(clip)
        nIn, nOut, W, H = self.nIn, self.nOut, self.W, self.H
        lead = nIn - nOut
        cap = cv2.VideoCapture(clip)
        t0f = (min(max(from_frame, 0), max(0, n_total - nIn)) // nOut) * nOut
        cap.set(cv2.CAP_PROP_POS_FRAMES, t0f)
        full, small = {}, {}
        state = dict(next=t0f, ended=False)

        def fetch_upto(need):
            while not state["ended"] and state["next"] <= need:
                t = now()
                ok, bgr = cap.read()
                if not ok:
                    state["ended"] = True; break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                full[state["next"]] = rgb
                small[state["next"]] = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
                state["next"] += 1
                tm.add("decode", t)

        out = [None] * n_total
        src = [None] * n_total                                  # how each point was found
        st = dict(windows=0, refined=0, dropped=0, coasted=0, coast_tried=0)
        last_idx = prev_idx = -1
        chain = 0

        def anchor(idx):
            if idx - 1 < 0 or out[idx - 1] is None:
                return None
            run, k = 0, idx - 1
            while k >= 0 and out[k] is not None:
                run += 1; k -= 1
            if run < 3:
                return None
            last = out[idx - 1]
            pi = next((j for j in range(idx - 2, max(0, idx - 5) - 1, -1) if out[j] is not None), -1)
            if pi < 0:
                return last[0], last[1]
            dt = idx - 1 - pi
            return last[0] + (last[0] - out[pi][0]) / dt, last[1] + (last[1] - out[pi][1]) / dt

        def refine(frs, x, y, g):
            t = now()
            r = self.ref.refine(frs, x, y, g)
            tm.add("refine", t)
            return r

        t = t0f
        while t + nIn <= n_total:
            fetch_upto(min(n_total - 1, t + nIn))
            win = [small.get(i) for i in range(t, t + nIn)]
            if any(w is None for w in win):
                break
            tp = now()
            x = np.concatenate([w.transpose(2, 0, 1) for w in win], 0).astype(np.float32)[None] / 255.0
            heat = self.s1.run(None, {self.inName: x})[0][0]
            tm.add("stage1", tp)
            st["windows"] += 1
            sx, sy = sw / W, sh / H
            for k in range(nOut):
                idx = t + lead + k
                if idx >= n_total:
                    continue
                prevF, cur, nxt = full.get(idx - 1), full.get(idx), full.get(idx + 1)
                trio = None if (prevF is None or cur is None or nxt is None) else [prevF, cur, nxt]
                plane = heat[k]
                pi = int(np.argmax(plane)); peak = float(plane.flat[pi])
                if peak >= gate and trio is not None:
                    py_, px_ = divmod(pi, W)
                    r = refine(trio, px_ * sx, py_ * sy, refine_gate)
                    if r is not None:
                        st["refined"] += 1
                        if out[idx] is None or peak > out[idx][2]:
                            out[idx] = (r[0], r[1], peak); src[idx] = "detected"
                    else:
                        st["dropped"] += 1
                if bridge:
                    if out[idx] is not None:
                        prev_idx, last_idx = last_idx, idx
                    elif last_idx >= 0 and idx - last_idx <= coast and trio is not None:
                        l = out[last_idx]; gap = idx - last_idx; vx = vy = 0.0
                        if prev_idx >= 0 and last_idx > prev_idx and out[prev_idx] is not None:
                            dt = last_idx - prev_idx
                            vx = (l[0] - out[prev_idx][0]) / dt; vy = (l[1] - out[prev_idx][1]) / dt
                        st["coast_tried"] += 1
                        r = refine(trio, l[0] + vx * gap, l[1] + vy * gap, coast_gate)
                        if r is not None:
                            out[idx] = (r[0], r[1], r[2]); src[idx] = "coasted"; st["coasted"] += 1
                            prev_idx, last_idx = last_idx, idx
                elif out[idx] is None and chain < coast and trio is not None:
                    a = anchor(idx)
                    if a is not None:
                        st["coast_tried"] += 1
                        r = refine(trio, a[0], a[1], coast_gate)
                        if r is not None:
                            out[idx] = (r[0], r[1], r[2]); src[idx] = "coasted"; st["coasted"] += 1; chain += 1
                        else:
                            chain = 0
                    else:
                        chain = 0
                elif out[idx] is not None:
                    chain = 0
                if back and idx >= 3 and out[idx] is not None and out[idx - 1] is not None and out[idx - 2] is None:
                    fa, fb, fc = full.get(idx - 3), full.get(idx - 2), full.get(idx - 1)
                    if fa is not None and fb is not None and fc is not None:
                        ex = 2 * out[idx - 1][0] - out[idx][0]; ey = 2 * out[idx - 1][1] - out[idx][1]
                        st["coast_tried"] += 1
                        r = refine([fa, fb, fc], ex, ey, coast_gate)
                        if r is not None:
                            out[idx - 2] = (r[0], r[1], r[2]); src[idx - 2] = "back-filled"; st["coasted"] += 1
            for kk in [k for k in full if k < t]:
                full.pop(kk, None); small.pop(kk, None)
            t += nOut
        cap.release()
        ungated = list(out)
        t = now()
        gated = flight_gate(ungated, margin=0)
        tm.add("flight_gate", t)
        st["kept"] = sum(p is not None for p in gated)
        st["rejected_by_flight_gate"] = sum(p is not None for p in ungated) - st["kept"]
        return gated, ungated, src, st, (n_total, fps, sw, sh)


# FlightGate.kt ----------------------------------------------------------------------------------
MAX_GAP, V_MAX, JUMP_K, JUMP_FLOOR, KEEP_FRAC = 8, 160.0, 3.0, 30.0, 0.35


def _step(p, a, b):
    return math.hypot(p[b][0] - p[a][0], p[b][1] - p[a][1]) / max(b - a, 1)


def _path_len(p, run):
    return sum(math.hypot(p[run[i]][0] - p[run[i - 1]][0], p[run[i]][1] - p[run[i - 1]][1])
               for i in range(1, len(run)))


def _split_jumps(p, run):
    if len(run) < 4:
        return [run]
    sp = [_step(p, run[i], run[i + 1]) for i in range(len(run) - 1)]
    lim = max(JUMP_K * sorted(sp)[len(sp) // 2], JUMP_FLOOR)
    out, cur = [], [run[0]]
    for n in range(1, len(run)):
        if sp[n - 1] > lim:
            out.append(cur); cur = [run[n]]
        else:
            cur.append(run[n])
    out.append(cur)
    return out


def kept_runs(p, keep_frac=KEEP_FRAC):
    idx = [i for i, q in enumerate(p) if q is not None]
    if len(idx) < 3:
        return []
    runs, cur = [], [idx[0]]
    for n in range(1, len(idx)):
        a, b = idx[n - 1], idx[n]
        if b - a <= MAX_GAP and _step(p, a, b) <= V_MAX:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    split = [r for run in runs for r in _split_jumps(p, run) if len(r) >= 3]
    if not split:
        return []
    lens = [_path_len(p, r) for r in split]
    bi = int(np.argmax(lens))
    return [r for i, r in enumerate(split) if i == bi or lens[i] >= keep_frac * lens[bi]]


def flight_gate(p, margin=0):
    keep = kept_runs(p)
    if not keep:
        return list(p)
    ok = [False] * len(p)
    for r in keep:
        for i in range(max(0, r[0] - margin), min(len(p) - 1, r[-1] + margin) + 1):
            ok[i] = True
    return [q if ok[i] else None for i, q in enumerate(p)]


# ANALYSES on the track ----------------------------------------------------------------------------
def track_2d_stats(pts, fps):
    idx = [i for i, p in enumerate(pts) if p is not None]
    if not idx:
        return dict(points=0)
    xs = np.array([pts[i][0] for i in idx]); ys = np.array([pts[i][1] for i in idx])
    span = idx[-1] - idx[0] + 1
    steps = [math.hypot(pts[b][0] - pts[a][0], pts[b][1] - pts[a][1]) / (b - a) for a, b in zip(idx, idx[1:])]
    bounce = None
    if len(idx) >= 5:
        vy = np.diff(ys) / np.maximum(np.diff(idx), 1)
        for j in range(1, len(vy)):
            if vy[j - 1] > 0 and vy[j] < 0:
                bounce = idx[j]; break
    return dict(points=len(idx), first_frame=idx[0], last_frame=idx[-1],
                visible_s=round(span / fps, 3), completeness=round(len(idx) / span, 3),
                px_per_frame_median=round(float(np.median(steps)), 1) if steps else None,
                image_bounce_frame=bounce,
                start_xy=[round(float(xs[0])), round(float(ys[0]))], end_xy=[round(float(xs[-1])), round(float(ys[-1]))])


def arc_3d(clip, ungated, fps, sw, release_frame, pitch_len, tm):
    """Camera from the two stump sets, then the gravity fit (arcfit.py = the Python mirror of Arc3d.kt)."""
    try:
        import pitchcam
        import arcfit
    except Exception as e:                                       # noqa: BLE001
        return dict(error=f"3D fit unavailable: {e}")
    t = now()
    Pm, pair, _ = pitchcam.camera_for_clip(clip, fracs=(0.0, 0.25, 0.5, 0.75, 0.98), L=pitch_len, conf=0.2)
    tm.add("pitch_lock", t)
    if Pm is None:
        return dict(pitch_lock=False, error="no stump set found at BOTH ends -> no metric speed / length / line")
    pts = {i: (p[0], p[1]) for i, p in enumerate(ungated) if p is not None}
    t = now()
    Q, r, k = arcfit.solve_best(Pm, pts, fps, L=pitch_len, release_frame=release_frame, src_w=sw)
    tm.add("arc_fit", t)
    if r is None:
        return dict(pitch_lock=True, solved=False, error="3D arc fit did not converge (too few ball points?)")
    ok, why = arcfit.plausible(r, fps, sw, pitch_len)
    _, _, info = arcfit.world_path(Q, r, fps, len(ungated) + 200)
    res = dict(pitch_lock=True, solved=True, plausible=bool(ok), reject_reason=None if ok else why,
               speed_kmh=round(r["speed_kmh"], 1), fit_rms_px=round(r["rms_px"], 2),
               points_used=len(r["used"]), vertical_scale=round(float(k), 3))
    bo = info.get("bounce_out")
    if bo is not None:
        res["length_m_from_batter_stumps"] = round(bo, 2)
        res["length_zone"] = ("yorker" if bo < 2 else "full" if bo < 5 else "good" if bo < 8 else "short")
    cr = info.get("cross")
    if cr is not None:
        x, z = cr
        res["at_stumps_line_m"] = round(x, 3)
        res["line"] = "stump line" if abs(x) <= 0.1143 else ("outside, +x side" if x > 0 else "outside, -x side")
        res["at_stumps_height_m"] = round(z, 3)
        res["hitting_stumps"] = bool(abs(x) <= 0.1143 + 0.036 and z <= 0.711 + 0.036)
    return res


def batting_cards(gated, tm):
    """BattingCards.kt trackSequence -> Contact (Hit/Miss), Slog Type (Lofted/Grounded) on a Hit."""
    f = [i for i, p in enumerate(gated) if p is not None]
    if len(f) < 12:
        return dict(contact=None, note="needs >= 12 tracked ball points")
    x = np.array([gated[i][0] for i in f], np.float32); y = np.array([gated[i][1] for i in f], np.float32)
    f = np.array(f, np.float32)
    dt = np.maximum(np.diff(f), 1)
    dx = np.diff(x) / dt; dy = np.diff(y) / dt; step = np.hypot(dx, dy)
    m = len(dx)
    sm = np.array([((dy[i - 1] if i > 0 else 0) + dy[i] + (dy[i + 1] if i < m - 1 else 0)) / 3 for i in range(m)])
    d = np.abs(np.sign(sm[1:]) - np.sign(sm[:-1]))
    bi = int(np.argmax(d)) if len(d) else 0
    if bi == 0:
        bi = len(f) // 3
    head = step[:min(max(bi, 2), m)]
    med = float(np.median(head)) if len(head) else 0.0
    med = med or 1.0
    rx = float(x.max() - x.min()) or 1.0; ry = float(y.max() - y.min()) or 1.0
    feat = np.stack([dx / med, dy / med, step / med, (x[:-1] - x[0]) / rx, (y[:-1] - y[0]) / ry], 1)
    span = max(f[m - 1] - f[0], 1)
    t0 = (f[:m] - f[0]) / span
    tq = np.arange(32) / 31.0
    seq = np.stack([np.interp(tq, t0, feat[:, c]) for c in range(5)], 1).astype(np.float32)[None]
    t = now()
    cp = float(onnx("batting_contact.onnx").run(None, {"track": seq})[0][0])
    res = dict(contact="Hit" if cp > 0.5 else "Miss", contact_conf=round(cp if cp > 0.5 else 1 - cp, 3), contact_p_hit=round(cp, 3))
    if cp > 0.5:
        sp = float(onnx("batting_slog.onnx").run(None, {"track": seq})[0][0])
        res.update(slog="Lofted" if sp > 0.5 else "Grounded", slog_conf=round(sp if sp > 0.5 else 1 - sp, 3))
    tm.add("batting", t)
    return res


# ------------------------------------------------------------------------------------------------
# CLIP ACTION (ClipActionConfirmer.kt): 3 frames (c-5, c, c+5) at 288x288, RAW 0-255 -> P(delivery)
# ------------------------------------------------------------------------------------------------
class ClipAction:
    S, STEP, STRIDE = 288, 5, 3

    def __init__(self):
        self.s = onnx("clip_action.onnx"); self.name = self.s.get_inputs()[0].name

    def score(self, path, tm):
        cap = cv2.VideoCapture(path); frames = []
        t = now()
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (self.S, self.S)).transpose(2, 0, 1).astype(np.float32))
        cap.release(); tm.add("decode_and_resize", t)
        curve = []; t = now()
        for c in range(self.STEP, len(frames) - self.STEP, self.STRIDE):
            x = np.concatenate([frames[c - self.STEP], frames[c], frames[c + self.STEP]], 0)[None]
            curve.append((c, float(np.ravel(self.s.run(None, {self.name: x})[0])[0])))
        tm.add("inference", t)
        return curve, len(frames)


# ------------------------------------------------------------------------------------------------
# letterbox bars: a phone video saved into a square (black bars top/bottom or left/right) is not what
# the models were trained on; the bars are trimmed before inference and results shifted back.
# ------------------------------------------------------------------------------------------------
def content_box(im, thr=12):
    g = im.max(2)
    r = np.where(g.mean(1) > thr)[0]; c = np.where(g.mean(0) > thr)[0]
    if len(r) < 16 or len(c) < 16:
        return 0, 0, im.shape[1], im.shape[0]
    return int(c[0]), int(r[0]), int(c[-1]) + 1, int(r[-1]) + 1


def shift(items, dx, dy):
    if not dx and not dy:
        return items
    for d in items:
        if "box" in d:
            d["box"] = [d["box"][0] + dx, d["box"][1] + dy, d["box"][2] + dx, d["box"][3] + dy]
        if d.get("xy") is not None:
            d["xy"] = [d["xy"][0] + dx, d["xy"][1] + dy]
        if "keypoints" in d:
            d["keypoints"] = {k: [v[0] + dx, v[1] + dy, v[2]] for k, v in d["keypoints"].items()}
    return items
