"""Recover the camera for an eval clip exactly the way the APP does.

The point of doing it this way rather than with the umpire-demo's own stump clustering is that the
eval has to answer a question about what SHIPS. The app runs `stump.tflite` on the clip's first
frame, splits the boxes into a near set and a far set about their midpoint, unions each side into one
wicket box, and takes the wicket half-width count-awarely - from the box width when the box is wide
enough to hold three stumps, from the box height when it is a single stump. Arc3d then builds the
projection from those eight corners and rescales the Z column against the near box. Every one of
those steps is reproduced here, so a number measured in this harness is a number the phone can
reach. (See ai/insights.md: an eval that infers something the deployment gets for free, or vice
versa, measures a pipeline nobody runs.)
"""
import os
import cv2
import numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:                      # older Pi images: pip install tflite-runtime
    from tflite_runtime.interpreter import Interpreter

PITCH_LEN = 20.12
STUMP_HALF = 0.1143
STUMP_H = 0.711
STUMP_ASPECT = STUMP_HALF / STUMP_H        # 0.1607 - the wicket's own proportions

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STUMP_TFLITE = os.environ.get("STUMP_TFLITE", os.path.join(_REPO, "app/app/src/main/assets/stump.tflite"))

_it = None


def _model():
    global _it
    if _it is None:
        _it = Interpreter(model_path=STUMP_TFLITE)
        _it.allocate_tensors()
    return _it


def detect(im, conf=0.25):
    """Stump boxes in NORMALISED frame coordinates: [(l, t, r, b, score), ...]."""
    it = _model()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    S = int(inp["shape"][2])
    h, w = im.shape[:2]
    s = min(S / w, S / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np.full((S, S, 3), 114, np.uint8)
    px, py = (S - nw) // 2, (S - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (nw, nh))
    x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    it.set_tensor(inp["index"], x)
    it.invoke()
    r = it.get_tensor(out["index"])[0]          # (4 + nc, anchors), raw - no NMS in the export
    boxes = r[:4].T
    # The export emits boxes NORMALISED to the letterbox, not in its pixels. Reading them as pixels
    # puts every wicket in the top-left 1/640th of the frame and no pair is ever found.
    if float(np.max(np.abs(boxes))) <= 1.5:
        boxes = boxes * S
    scores = r[4:].max(0)
    cls = r[4:].argmax(0)
    keep = (scores >= conf) & (cls == 0)        # class 0 = stumps, class 1 = bails
    dets = []
    for (cx, cy, bw, bh), sc in zip(boxes[keep], scores[keep]):
        l, t_, rr, b = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        dets.append((((l - px) / s) / w, ((t_ - py) / s) / h,
                     ((rr - px) / s) / w, ((b - py) / s) / h, float(sc)))
    return _nms(dets)


def _nms(d, thr=0.45):
    d = sorted(d, key=lambda x: -x[4])
    out = []
    for b in d:
        if all(_iou(b, k) < thr for k in out):
            out.append(b)
    return out


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def pitch_pair(dets):
    """The app's PitchOverlay.pitchPairOf: split about the midpoint, union each side."""
    if len(dets) < 2:
        return None
    cy = [(d[1] + d[3]) / 2 for d in dets]
    if max(cy) - min(cy) < 0.06:
        return None                              # stumps do not span two ends
    mid = (max(cy) + min(cy)) / 2
    far = [d for d, c in zip(dets, cy) if c < mid]
    near = [d for d, c in zip(dets, cy) if c >= mid]
    if not far or not near:
        return None
    def union(g):
        return (min(x[0] for x in g), min(x[1] for x in g),
                max(x[2] for x in g), max(x[3] for x in g))
    n, f = union(near), union(far)
    if (f[3] - f[1]) > (n[3] - n[1]) * 1.15:
        return None                              # the far wicket cannot be the taller one
    return n, f


def half_norm(box, ar):
    """Wicket half-width as a fraction of frame width - count-aware, as the app does it."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    aspect = w / (h * ar) if h > 0 else 0.0
    return w / 2 if aspect >= 0.18 else STUMP_ASPECT * h * ar


def homography(world, img):
    A, b = [], []
    for (X, Y), (u, v) in zip(world, img):
        A.append([X, Y, 1, 0, 0, 0, -X * u, -Y * u]); b.append(u)
        A.append([0, 0, 0, X, Y, 1, -X * v, -Y * v]); b.append(v)
    try:
        h = np.linalg.solve(np.array(A, float), np.array(b, float))
    except np.linalg.LinAlgError:
        return None
    return np.append(h, 1.0).reshape(3, 3)


def camera(near, far, W, H, L=PITCH_LEN):
    """Full 3x4 projection from the two wicket boxes, plus the near-box height calibration.

    [L] is the distance between the two detected wickets. It is 20.12 m on a cricket pitch and NOT
    on a practice net, where the far "wicket" can be a target stump part-way down the mat. Getting
    it wrong compresses every distance, every speed and the whole flight by the same factor, and
    nothing in the camera's own sanity checks can see it - the solved pose stays perfectly plausible.
    """
    ar = H / W
    nhalf = half_norm(near, ar) * W
    fhalf = half_norm(far, ar) * W
    ncx, nb = (near[0] + near[2]) / 2 * W, near[3] * H
    fcx, fb = (far[0] + far[2]) / 2 * W, far[3] * H
    ftop = far[1] * H
    if nhalf < 2 or fhalf < 0.5 or nb - fb < 20 or fb - ftop < 4:
        return None
    # The near wicket is ALWAYS the bowler's: every clip is shot from the umpire's view, standing at
    # the bowler's end (David, 2026-08-21: "we dont record behind the batsman at all"). Solving for
    # which end is which was tried and removed - it is a degree of freedom the data does not need,
    # and on a weak track it happily returns a back-to-front world that fits better than the truth.
    hg = homography([(-STUMP_HALF, 0.0), (STUMP_HALF, 0.0), (-STUMP_HALF, L), (STUMP_HALF, L)],
                    [(ncx - nhalf, nb), (ncx + nhalf, nb), (fcx - fhalf, fb), (fcx + fhalf, fb)])
    hs = homography([(-STUMP_HALF, 0), (STUMP_HALF, 0), (-STUMP_HALF, STUMP_H), (STUMP_HALF, STUMP_H)],
                    [(fcx - fhalf, fb), (fcx + fhalf, fb), (fcx - fhalf, ftop), (fcx + fhalf, ftop)])
    if hg is None or hs is None:
        return None
    p1, p2, p4 = hg[:, 0], hg[:, 1], hg[:, 2]
    sx, sz = hs[:, 0], hs[:, 1]
    j = int(np.argmax(np.abs(p1)))
    if abs(p1[j]) < 1e-12:
        return None
    p3 = sz / (sx[j] / p1[j])
    # heights from the NEAR box: the far box is a few dozen pixels tall, so a pixel of error there is
    # several percent of stump height here
    det_h = (near[3] - near[1]) * H
    target = nb - det_h
    den = STUMP_H * (p3[1] - target * p3[2])
    if abs(den) > 1e-9:
        k = (target * p4[2] - p4[1]) / den
        if np.isfinite(k) and 0.25 <= k <= 3.0:
            p3 = p3 * k
    return np.stack([p1, p2, p3, p4], 1)


# MEASURED STUMP-TO-STUMP DISTANCE PER VENUE, where we have actually been told it.
# David, 12 Sep 2026: the Gopalan Hoodi OUTDOOR net is "exactly 20.5m ... stump to stump" - it was
# built 38 cm long by mistake. Regulation is 20.12 m, and `camera()`'s own docstring warns that a
# wrong L "compresses every distance, every speed and the whole flight by the same factor, and
# nothing in the camera's own sanity checks can see it". Assuming 20.12 at a 20.5 m venue
# UNDER-states every speed there by 20.12/20.5 = 1.9%.
# Venues not listed fall back to regulation because nobody has measured them - that is an
# admission, not a claim. The Hoodi INDOOR hall (venue 08) carries most of our corpus and its real
# length is still unknown; it is worth a tape measure.
VENUE_PITCH_M = {
    "09": 20.5,      # Gopalan Hoodi Outdoor - MEASURED (built 38 cm long by mistake)
    "08": 20.12,     # Gopalan Hoodi Indoor  - MEASURED, and it is regulation (David, 12 Sep)
    "23": 20.12,     # LSG Chennai - a professional pitch built to regulation (David, 2 Oct 2026)
}

# Venue 08 is listed above even though its value equals the fallback, deliberately: "measured and
# happens to be regulation" is a different claim from "never measured, assumed regulation", and
# venue 08 carries the majority of the whole corpus. Anything NOT in this map is still an
# assumption.


def pitch_len_for_venue(venue, default=PITCH_LEN):
    return VENUE_PITCH_M.get(str(venue), default)


def camera_for_clip(path, fracs=(0.0, 0.5, 0.98), L=PITCH_LEN, conf=0.25):
    """Try the first, middle and last frame - one frame does not always show both wickets.

    [L] is the venue's real stump-to-stump distance; see VENUE_PITCH_M.

    [conf] is the stump detector's threshold. The default 3-frame / 0.25 search is FAST, not
    thorough: measured on a 24-clip stratified sample of the pixel key it solves 19/24, while
    sweeping 13 frames at conf 0.15 solves 23/24 - and the sane-camera filter rejects none of
    either, so the difference is purely how hard we looked. Callers that care about coverage
    rather than speed should pass the wider search.
    """
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for fr in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, min(n - 1, fr * n))))
        ok, im = cap.read()
        if not ok:
            continue
        pair = pitch_pair(detect(im, conf=conf))
        if pair is None:
            continue
        Pm = camera(pair[0], pair[1], W, H, L=L)
        if Pm is not None:
            cap.release()
            return Pm, pair, (W, H)
    cap.release()
    return None, None, (W, H)
