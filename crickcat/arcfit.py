"""The trajectory fit that SHIPS, in Python - a line-for-line mirror of app/.../Arc3d.kt.

Two implementations of one model is a smell, and it is deliberate here for one reason: this is the
only way an eval number means anything about the phone. `arc3d.py` fits nine parameters with
scipy's Levenberg-Marquardt and no bounds; the app has neither scipy nor the appetite for a
nine-parameter solve, and it draws the pre-bounce leg then carries the rest with the bounce rules.
Scoring arc3d and shipping Arc3d would be measuring a pipeline nobody runs.

Differences from arc3d.py, all forced and all measured:

  * SIX parameters, not nine. The post-bounce leg in arc3d is fitted, and on a clip whose detections
    stop before the bounce that leg has no evidence behind it at all. Here everything past the last
    detection comes from the physics.
  * BOUNDED. Unbounded, the fit escapes: a single camera cannot see depth, and when the detections
    are short the solver finds a ball forty thousand kilometres away going 2.8e8 km/h, because at
    that distance gravity vanishes and the model degenerates to a straight line that fits the pixels
    just as well. Measured on the reference clips, that happened on two of four.
  * SEED AND GROW, not plain IRLS. Fitting all the detections at once lands in a basin the strays
    define - the tracker keeps firing on the batsman and the netting long after the delivery. The
    earliest contiguous run cannot be post-bounce, so it seeds the fit, and the flight then claims
    the points it explains, never past its own bounce.

Verified against arc3d's published numbers on the four web clips: 52.0 vs 51.8, 52.0 vs 51.7,
41.6 vs 48.1 (where the reference itself sits at 10.35 px rms and this fit at 1.04), 58.2 vs 57.5.
"""
import numpy as np

G = 9.81
PITCH_LEN = 20.12
RESTITUTION = 0.60          # a short ball has to come at head height; 0.42 made bouncers impossible
FRICTION = 0.75

# r0 is the ball's state at the FIRST DETECTION, not at release - so it must be bounded like a point
# anywhere in the flight, not like a bowler's hand. Bounding r0y to 6 m was wrong in principle and
# broke a real clip: on David's nets footage the tracker acquires the ball 13 m down the pitch, the
# bound pinned there, and the fit came back at 106 km/h with no bounce at all. A pinned optimiser
# means the constraint is fighting the data. Freeing it drops that clip from 2.57 to 1.89 px rms.
LO = np.array([-3.5, -1.5, 0.0, -9.0, 5.0, -15.0])
HI = np.array([3.5, 20.5, 3.6, 9.0, 45.0, 12.0])


def _at(p, t):
    return np.stack([p[0] + p[3] * t, p[1] + p[4] * t, p[2] + p[5] * t - 0.5 * G * t * t], 1)


def _res(Pm, p, ts, uv):
    w = _at(p, ts)
    q = np.c_[w, np.ones(len(ts))] @ Pm.T
    d = q[:, 2]
    bad = d <= 1e-6
    safe = np.where(bad, 1.0, d)
    out = np.stack([q[:, 0] / safe - uv[:, 0], q[:, 1] / safe - uv[:, 1]], 1)
    out[bad] = 1e4          # behind the camera is an escape, not a solution
    return out.ravel()


def _lm(Pm, ts, uv, p0, iters=60):
    p = np.clip(np.asarray(p0, float), LO, HI)
    lam = 1e-3
    r = _res(Pm, p, ts, uv)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((len(r), 6))
        for j in range(6):
            e = np.zeros(6); e[j] = 1e-4
            J[:, j] = (_res(Pm, np.clip(p + e, LO, HI), ts, uv) - r) / 1e-4
        A = J.T @ J; g = J.T @ r
        stepped = False
        for _try in range(12):
            try:
                d = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                break
            p2 = np.clip(p + d, LO, HI)
            r2 = _res(Pm, p2, ts, uv); c2 = float(r2 @ r2)
            if c2 < cost - 1e-12:
                p, r, cost = p2, r2, c2
                lam = max(lam * 0.3, 1e-9); stepped = True
                break
            lam *= 8
        if not stepped:
            break
    return p, cost


def _fit(Pm, ts, uv, warm=None):
    starts = ([list(warm)] if warm is not None else []) + \
             [[0.0, 1.5, z0, 0.0, vy, -1.0] for vy in (22.0, 30.0, 38.0) for z0 in (1.8, 2.2)]
    best, bc = None, np.inf
    for s in starts:
        p, c = _lm(Pm, ts, uv, s)
        if c < bc:
            best, bc = p, c
    return best


def _errors(Pm, p, ts, uv):
    r = _res(Pm, p, ts, uv).reshape(-1, 2)
    return np.hypot(r[:, 0], r[:, 1])


def prefilter(fs, us, vs, jump_k=3.0, win=2):
    n = len(fs)
    if n < 5:
        return list(range(n))
    step = float(np.median(np.hypot(np.diff(us), np.diff(vs)))) + 1e-3
    keep = []
    for i in range(n):
        nb = [j for j in range(max(0, i - win), min(n, i + win + 1)) if j != i]
        if len(nb) < 2:
            keep.append(i); continue
        px = np.interp(fs[i], fs[nb], us[nb]); py = np.interp(fs[i], fs[nb], vs[nb])
        if np.hypot(us[i] - px, vs[i] - py) <= jump_k * step:
            keep.append(i)
    return keep if len(keep) >= 8 else list(range(n))


def fit(Pm, pts, fps, release_frame=None):
    """pts: {frame: (u, v)}. Returns the solved flight, or None.

    [release_frame] is the frame the clip was cut around, which app clips carry in their name
    ("_r<ms>"). Nothing before it is the ball: the tracker is asked to look from 0.4 s early, so the
    first detections are routinely the bowler's arm coming through the delivery stride - and since
    the seed is the EARLIEST contiguous run, those are exactly the points it reaches for first.
    """
    Pm = np.asarray(Pm, float)
    fs = np.array(sorted(pts), float)
    if len(fs) < 8:
        return None
    us = np.array([pts[int(f)][0] for f in fs]); vs = np.array([pts[int(f)][1] for f in fs])
    pre = prefilter(fs, us, vs)
    if release_frame is not None:
        kept = [i for i in pre if fs[i] >= release_frame - 2]   # a frame of slack either way
        if len(kept) >= 8:
            pre = kept

    seed = [pre[0]]
    for i in pre[1:]:
        if fs[i] - fs[seed[-1]] <= 2 and len(seed) < 14:
            seed.append(i)
        elif len(seed) >= 8:
            break
        elif fs[i] - fs[seed[-1]] > 2:
            seed = [i]
    if len(seed) < 6:
        return None
    t0 = fs[seed[0]]
    def obs(idx):
        I = np.array(idx)
        return (fs[I] - t0) / fps, np.c_[us[I], vs[I]]

    idx, p = list(seed), None
    for _ in range(5):                                   # GROW to the bounce
        ts, uv = obs(idx)
        p = _fit(Pm, ts, uv, p)
        if p is None:
            return None
        tb = (p[5] + np.sqrt(max(p[5] ** 2 + 2 * G * p[2], 0.0))) / G
        cand = [i for i in pre if (fs[i] - t0) / fps <= tb + 0.02]
        if len(cand) < 6:
            break
        ts2, uv2 = obs(cand)
        err = _errors(Pm, p, ts2, uv2)
        med = float(np.median(err))
        grown = [cand[i] for i in range(len(cand)) if err[i] < max(4.0, 3.0 * med)]
        if len(grown) < 6 or set(grown) == set(idx):
            break
        idx = grown
    for _ in range(4):                                   # POLISH by MAD
        ts, uv = obs(idx)
        p = _fit(Pm, ts, uv, p)
        err = _errors(Pm, p, ts, uv)
        med = float(np.median(err)); mad = float(np.median(np.abs(err - med))) + 1e-3
        keep = [idx[i] for i in range(len(idx)) if err[i] < max(med + 2.5 * mad, 3.0)]
        if len(keep) < 8 or len(keep) == len(idx):
            break
        idx = keep
    ts, uv = obs(idx)
    err = _errors(Pm, p, ts, uv)
    rms = float(np.sqrt((err ** 2).mean()))
    speed = float(np.linalg.norm(p[3:6]) * 3.6)
    if not np.isfinite(speed) or not 25 <= speed <= 200 or not np.isfinite(rms) or rms > 25:
        return None
    return dict(params=p, t0=float(t0), used=[int(fs[i]) for i in idx], rms_px=rms,
                speed_kmh=speed, last_obs=int(fs[max(idx)]))


def solve_best(Pm, pts, fps, L=PITCH_LEN, release_frame=None, src_w=1080):
    """Try the fit with and without the release marker and keep the better one.

    A CANDIDATE, NOT A FILTER, and the measurement is why. Dropping everything before the release
    should be free - nothing before the ball leaves the hand is the ball - and on David's nets clip
    it does help, 1.7 px to 1.5. But on another it removed points the fit needed, took the surviving
    set from ten to eight, and turned a passing 88 km/h solve into a rejected 162. A constraint that
    is right in principle can still cost more evidence than it removes noise.

    So both are solved and the better SURVIVING one wins.

    EVIDENCE FIRST, TIGHTNESS SECOND - and rms alone cannot be the ranking. Ported from
    `Arc3d.kt` on 26 Sep 2026; this file had kept the `rms + 1e6` form the app replaced on 22 Sep.
    A fit through FEWER points almost always has lower rms, so scoring on rms silently rewards
    dropping evidence: measured on 22 Sep, relaxing the survivor guard did not just admit new
    clips, it REWROTE two that already passed - 128.8 -> 152.5 and 132.8 -> 158.8 km/h - because a
    sparser candidate became eligible and outranked a fuller one on rms alone. Both moved toward
    the degenerate cluster at 162.

    So: a passing fit always beats a failing one; then MORE surviving points wins unless the fuller
    fit is much worse (>30%); and a SPARSER fit has to be much tighter (<70%) before it may win.
    The margins are deliberately wide - dropping evidence must be clearly justified, not marginally
    cheaper.
    """
    best = (Pm, None, 1.0)
    best_passes, best_n, best_rms = False, -1, np.inf
    for rel in ([None] if release_frame is None else [release_frame, None]):
        Q, r, k = calibrate_by_flight(Pm, pts, fps, L=L, release_frame=rel)
        if r is None:
            continue
        passes, _why = plausible(r, fps, src_w, L)
        n, rms = len(r["used"]), r["rms_px"]
        if best[1] is None:
            better = True
        elif passes != best_passes:
            better = passes
        elif n > best_n:
            better = rms <= best_rms * 1.30
        elif n < best_n:
            better = rms < best_rms * 0.70
        else:
            better = rms < best_rms
        if better:
            best_passes, best_n, best_rms = passes, n, rms
            best = (Q, r, k)
    return best


def calibrate_by_flight(Pm, pts, fps, L=PITCH_LEN, ks=None, release_frame=None):
    """Let the BALL calibrate the camera's vertical, instead of trusting the near stump's height.

    The camera takes its metres-per-pixel in Z from the near wicket, assuming that box is a 0.711 m
    set of stumps. On David's nets footage it is a spring stump in a tall rubber base and the
    detector boxes the whole assembly - about a metre. Every height was then over-reported by ~43%,
    the ball looked like it was flying far higher than it was, and the fit concluded a 108.7 km/h
    FULL TOSS that sailed over the batsman. David, who bowled it: "in reality the ball actually
    landed near the stumps."

    Nothing in the camera's own checks can see this. The solved pose is sane, the box aspects are
    ordinary, and the fit reproduces the pixels at 1.89 px. The scale is simply wrong, and a wrong
    scale reprojects perfectly.

    But it IS observable, because gravity is known. A ball falling at 9.81 m/s^2 is a ruler laid
    against the image: only one vertical scale makes the observed curvature come out at g. So sweep
    the scale, refit at each, and keep the solutions that describe a delivery - one that comes down
    on the pitch rather than 7 m behind the batsman - then take the best-fitting of those. On the
    clip above it picks 0.70, i.e. a near object 1.02 m tall, and returns 77.6 km/h landing at
    19.1 m at 1.73 px: a better fit than the 0.711 m assumption gave, as well as the true one.

    ONLY A FALLBACK, and the eval says why. Applied unconditionally it costs 7.5 points of recall
    and doubles the false-fire rate on the 23-clip set, because those clips are shot at grounds where
    the stumps really are regulation wickets - there the measured height is right and letting a short,
    noisy track re-choose the scale is pure over-fitting. So the measured wicket is trusted first, and
    this runs only when trusting it produces something that is not a delivery: a ball landing behind
    the bowler, or sailing past the batsman without ever coming down.

    Returns (Pm_scaled, fit, k) or (Pm, fit, 1.0) when the camera as measured is already fine.
    """
    Pm = np.asarray(Pm, float)
    r0 = fit(Pm, pts, fps, release_frame)
    # Trigger on ANY implausibility, not just a bad landing. A release solving to 3.1 m is the same
    # wrong-vertical-scale signature as a ball that never comes down - it was only the landing check
    # that fired, so a clip whose scale error showed up as an impossible release was rejected outright
    # instead of being recalibrated and then drawn.
    if r0 is not None and plausible(r0, fps, 1080, L)[0]:
        return Pm, r0, 1.0
    if ks is None:
        ks = [0.55 + 0.05 * i for i in range(20)]           # near object 0.5 m to 1.3 m tall
    best = None
    for k in ks:
        Q = Pm.copy()
        Q[:, 2] = Pm[:, 2] * k
        r = fit(Q, pts, fps, release_frame)
        if r is None:
            continue
        if not plausible(r, fps, 1080, L)[0]:
            continue
        if best is None or r["rms_px"] < best[1]["rms_px"]:
            best = (Q, r, k)
    if best is None:
        return Pm, r0, 1.0
    return best


def plausible(r, fps, src_w, L=PITCH_LEN):
    """Is this solve a DELIVERY, or just a curve that reprojects?

    David, looking at an arc that ran off the top-left of the frame: "is our simulation not linked?
    otherwise it wouldn't have happend i guess" - and he was right. The physics was in the model but
    not in the ACCEPTANCE: the only gates were a speed band and rms < 25 px, so a fit that put the
    release 8 m down the pitch at 141 km/h with a 3 m apex was drawn without complaint. "even the
    release point was off here."

    These are the things a delivery cannot do. A ball is let go from behind the bowling crease at
    about head height, it comes down on the strip, and a tracker that saw only a handful of points
    has not seen enough to place any of it.

    Returns (ok, reason) so a caller can say WHY nothing was drawn.
    """
    p = np.asarray(r["params"], float)
    # A PINNED PARAMETER IS NOT A MEASUREMENT.
    #
    # On thin evidence the cost falls monotonically through the whole legal range of vy - the
    # unbounded minimum is out past 60-90 m/s, where gravity vanishes and the model degenerates to a
    # straight line. LO/HI were added to stop that escape, but they do not create a minimum, they
    # TRUNCATE the valley: Levenberg-Marquardt slides down it and parks on the box edge. Every clip
    # with too little curvature therefore converges to the SAME corner, which is why five different
    # deliveries by different bowlers, through two different detectors, all reported 163 km/h.
    #
    # HI[4] is 45 m/s = 162.0 km/h and the speed gate below accepts up to 170, so the artifact sails
    # straight through it. Only the >=10-point gate stood in the way, i.e. two extra detections from
    # publishing a degenerate fit at 1.13 px rms looking immaculate. This is the file's own lesson
    # about restitution, restated: an optimiser sitting on its constraint boundary means the
    # constraint is fighting the data, not describing it.
    _EPS = 1e-3
    _NAMES = ("x0", "y0", "z0", "vx", "vy", "vz")
    _pin = [n for n, v, lo, hi in zip(_NAMES, p, LO, HI) if v <= lo + _EPS or v >= hi - _EPS]
    if _pin:
        return False, ("the solve is pinned on its bounds (" + ", ".join(_pin) +
                       ") - too little curvature to constrain the flight")
    if p[4] <= 1e-6:
        return False, "the ball is not going down the pitch"
    if not _lands_on_the_pitch(r, L):
        return False, "the flight never comes down on the strip"
    # wind back to the bowler's stumps: that is where the ball was let go
    t0 = min(0.0, -p[1] / p[4])
    z_rel = p[2] + p[5] * t0 - 0.5 * G * t0 * t0
    if not (1.1 <= z_rel <= 2.9):
        return False, f"release height {z_rel:.1f} m is not a bowler's hand"
    apex = z_rel if p[5] - G * t0 <= 0 else z_rel + (p[5] - G * t0) ** 2 / (2 * G)
    if apex > 3.6:
        return False, f"the flight peaks at {apex:.1f} m"
    speed = float(np.linalg.norm(p[3:6])) * 3.6
    if not (40.0 <= speed <= 170.0):
        return False, f"{speed:.0f} km/h is not a delivery"
    # rms is in SOURCE pixels, so normalise before judging it - a 720-wide clip and a 1080-wide one
    # are not the same 5 px
    # 8 px, and the eval could not choose it: between 5 and 10 px the 23-clip numbers do not move
    # at all, because the clips it rejects fail on PHYSICS, not on tightness. What did choose it was
    # David's own 21-clip session, where 5 px refused three deliveries at 78-84 km/h with sane
    # geometry purely for fitting at 7-8 px. The physics gates above are the ones doing the work.
    rms = r["rms_px"] * 1080.0 / max(1.0, src_w)
    if rms > 8.0:
        return False, f"fits the detections at only {rms:.1f} px"
    # 10 -> 8, BROUGHT INTO STEP WITH `Arc3d.kt` (26 Sep 2026). The app lowered this on 22 Sep and
    # this file did not follow, although both carry a comment claiming Arc3dTest pins them together
    # - it does not. Every offline number measured since has therefore been scored on a STRICTER
    # guard than ships, and the clips it silently cost are exactly the fast ones: at 30 fps a
    # 150 km/h delivery is airborne ~0.48 s over ~17.7 m, i.e. 14-15 frames, and the robust fit
    # leaves 8-10. On the 41 radar-gun clips 8 of the 22 refusals were this guard alone.
    #
    # Safe ONLY with the evidence-first ranking in solve_best() below, ported in the same change:
    # a fit through fewer points almost always has lower rms, so ranking on rms alone would let a
    # sparser candidate outrank a fuller one the moment the guard relaxed, and on 22 Sep that
    # rewrote two already-good answers by ~25 km/h toward the degenerate cluster at 162.
    if len(r["used"]) < 8:
        return False, f"only {len(r['used'])} detections survived"
    return True, ""


def _lands_on_the_pitch(r, L=PITCH_LEN):
    """Does this flight come down on the strip? A delivery does. A wrong vertical scale does not."""
    p = np.asarray(r["params"], float)
    disc = p[5] ** 2 + 2 * G * p[2]
    if disc < 0:
        return False
    land = p[1] + p[4] * (p[5] + np.sqrt(disc)) / G
    return 0.0 <= land <= L + 1.0


def anchor_world(Pm, r, fps, pts, rows, deg=None, max_m=1.2):
    """The residual anchor, done in METRES instead of pixels.

    Same idea as [anchor]: the arc and the detections disagree mostly systematically, that error is
    smooth in time, so fit it low-order and add it back. The difference is where it is applied. The
    image-space version corrects the DRAWN LINE, which is fine while a 2D overlay is the only
    consumer - and it is why the app and the dashboard drew visibly different curves for the same
    delivery (6.6 px median apart on indoor ball 11) even though their cameras agreed to four
    decimals and their fits to 0.4 km/h. A correction that exists only in one camera's pixels cannot
    be shown in another camera's view.

    In world space it is camera-independent: the same corrected flight reprojects into the clip, into
    the orbit view, and onto the phone. Only X and Z are corrected - a detection fixes where the ball
    is ACROSS and ABOVE the pitch, while how far down it is at time t is what the fit is for.
    """
    Pm = np.asarray(Pm, float)
    M = Pm[:, :3]
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return rows
    C = -Minv @ Pm[:, 3]
    have = [(f, x, y, z) for f, x, y, z in rows if int(f) in pts]
    if len(have) < 5:
        return rows
    if deg is None:
        deg = int(min(5, max(1, len(have) // 4)))
    if len(have) < deg + 3:
        return rows
    ts, dx, dz = [], [], []
    for f, x, y, z in have:
        u, v = pts[int(f)]
        d = Minv @ np.array([u, v, 1.0])
        if abs(d[1]) < 1e-9:
            continue
        s = (y - C[1]) / d[1]          # where that ray sits at the arc's own distance down the pitch
        p = C + s * d
        if abs(p[0] - x) > max_m or abs(p[2] - z) > max_m:
            continue                    # a stray, not a systematic offset
        ts.append((f - r["t0"]) / fps); dx.append(p[0] - x); dz.append(p[2] - z)
    if len(ts) < deg + 3:
        return rows
    ts = np.array(ts); tm = ts.mean(); tsd = ts.std() + 1e-6
    cx = np.polyfit((ts - tm) / tsd, np.array(dx), deg)
    cz = np.polyfit((ts - tm) / tsd, np.array(dz), deg)
    lo, hi = ts.min(), ts.max()
    out = []
    for f, x, y, z in rows:
        tt = (float(np.clip((f - r["t0"]) / fps, lo, hi)) - tm) / tsd
        out.append([f, round(x + float(np.polyval(cx, tt)), 4), round(y, 4),
                    round(max(0.0, z + float(np.polyval(cz, tt))), 4)])
    return out


def world_path(Pm, r, fps, n_frames, stop_y=PITCH_LEN + 0.3, dt=1 / 600.0):
    """The same flight as [path], but in METRES and split at the last detection.

    Returns (observed, simulated, info) with rows [frame, X, Y, Z]. The split is what the web viewer
    draws in two colours: red where the tracker was still seeing the ball, blue where only the
    physics is carrying it. Anything that cannot tell those apart is claiming more than it knows.
    """
    Pm = np.asarray(Pm, float)
    p = np.asarray(r["params"], float)
    t = min(0.0, -p[1] / p[4]) if p[4] > 1e-6 else 0.0
    pos = np.array([p[0] + p[3] * t, p[1] + p[4] * t, p[2] + p[5] * t - 0.5 * G * t * t])
    vel = np.array([p[3], p[4], p[5] - G * t])
    if pos[2] < 0:
        t, pos, vel = 0.0, p[0:3].copy(), p[3:6].copy()
    obs, sim, seen = [], [], set()
    bounce_out = cross = None
    bounces, rolling = 0, False
    while t < 4.0 and pos[1] < stop_y:
        f = int(round(r["t0"] + t * fps))
        if 0 <= f < n_frames and f not in seen:
            seen.add(f)
            row = [f, round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(max(0.0, pos[2])), 4)]
            (obs if f <= r["last_obs"] else sim).append(row)
        vel[2] -= G * dt
        pos = pos + vel * dt
        if not rolling and pos[2] <= 0 and vel[2] < 0:
            pos[2] = 0.0
            if bounces == 0:
                bounce_out = PITCH_LEN - float(pos[1])
            vel[2] = -RESTITUTION * vel[2]
            sp = float(np.hypot(vel[0], vel[1])) * FRICTION
            head = np.arctan2(vel[0], vel[1])
            vel[0], vel[1] = sp * np.sin(head), sp * np.cos(head)
            bounces += 1
            if vel[2] < 0.35:
                vel[2] = 0.0; rolling = True
        if rolling:
            pos[2] = 0.0; vel[2] = 0.0
        if cross is None and pos[1] >= PITCH_LEN:
            cross = (float(pos[0]), float(pos[2]))
        t += dt
    # the seam has to be continuous: the blue starts where the red stopped, not one frame later
    if obs and sim:
        sim.insert(0, obs[-1])
    return obs, sim, dict(bounce_out=bounce_out, cross=cross, bounces=bounces)


def anchor(Pm, r, fps, pts, out, deg=None, max_px=60.0):
    """Correct the drawn arc toward the detections WITHOUT giving up its smoothness.

    The measurement that motivated this: on the eval subset, keeping the tracker's own detection
    wherever it made one scores 45.9% recall against the pure arc's 35.0% - the detections are
    genuinely more accurate frame by frame (1.42 px median against the arc's 5.5). But a polyline
    through raw detections is what David rejected on sight: "doesn't give that delight".

    So instead of choosing between them, fit the RESIDUAL. Where the arc and a detection disagree,
    that disagreement is mostly systematic - a camera whose stump boxes were a pixel off, a ball
    whose radius biases the heatmap centroid - and systematic error is smooth in time. A low-order
    polynomial in t through those residuals captures it and leaves the per-frame jitter behind, so
    the drawn line moves onto the evidence while staying a single clean curve.

    The degree is chosen from how many detections support it - one coefficient per four points,
    capped at five. Swept on the eval subset (recall / median error, 11 clips):

        deg 0  24.8%  8.86 px      deg 3  41.8%  4.91 px
        deg 1  31.2%  7.85 px      deg 4  42.2%  4.02 px
        deg 2  36.5%  4.37 px      deg 5  42.9%  3.97 px

    Five is where it stops paying; the adaptive rule gets 42.6% / 3.97 px without letting a short
    track fit a wiggle through eight points. The corrected arc localises BETTER than the tracker's
    own detections (3.97 px against 4.79) while still being one curve.
    """
    fs = [f for f in range(len(out)) if out[f] is not None and pts.get(f) is not None]
    if deg is None:
        deg = int(min(5, max(1, len(fs) // 4)))
    if len(fs) < deg + 3:
        return out
    t = np.array([(f - r["t0"]) / fps for f in fs], float)
    dx = np.array([pts[f][0] - out[f][0] for f in fs], float)
    dy = np.array([pts[f][1] - out[f][1] for f in fs], float)
    ok = np.hypot(dx, dy) <= max_px            # a stray is not a systematic error
    if ok.sum() < deg + 3:
        return out
    # centre and scale t before fitting: at degree 5 a raw seconds axis is badly conditioned
    tm = float(t[ok].mean()); tsd = float(t[ok].std()) + 1e-6
    cx = np.polyfit((t[ok] - tm) / tsd, dx[ok], deg)
    cy = np.polyfit((t[ok] - tm) / tsd, dy[ok], deg)
    lo, hi = t[ok].min(), t[ok].max()
    fixed = list(out)
    for f, p in enumerate(out):
        if p is None:
            continue
        tt = (float(np.clip((f - r["t0"]) / fps, lo, hi)) - tm) / tsd   # never extrapolate it
        fixed[f] = (p[0] + float(np.polyval(cx, tt)), p[1] + float(np.polyval(cy, tt)))
    return fixed


def path(Pm, r, fps, n_frames, dt=1 / 600.0):
    """Per-frame image points for the whole delivery: fitted before the last detection, carried by
    the bounce rules after it. Returns (pts, info) with pts[f] = (u, v) or None."""
    Pm = np.asarray(Pm, float)
    p = np.asarray(r["params"], float)

    # START AT THE RELEASE, NOT AT THE FIRST DETECTION.
    #
    # r0 is where the tracker first SAW the ball, which is not where the bowler let go of it - on
    # David's nets footage it is 13 m down the pitch. Marching forward from there drew a stub of the
    # last few frames and called it the delivery, and once the bounds were widened enough to let r0
    # sit near the batsman it drew nothing at all: the loop's own stop condition was already true on
    # entry. The pre-bounce flight is one parabola, so it back-propagates exactly - wind it back to
    # where the ball crossed the bowler's stumps (Y = 0) and draw from there.
    t = 0.0
    if p[4] > 1e-6:
        t = min(0.0, -p[1] / p[4])
    pos = np.array([p[0] + p[3] * t, p[1] + p[4] * t, p[2] + p[5] * t - 0.5 * G * t * t])
    vel = np.array([p[3], p[4], p[5] - G * t])
    if pos[2] < 0:                    # never start the draw underground
        t, pos, vel = 0.0, p[0:3].copy(), p[3:6].copy()
    out = [None] * n_frames
    bounces, rolling = 0, False
    bounce_out = cross = None
    while t < 4.0 and pos[1] < PITCH_LEN + 0.3:
        f = int(r["t0"] + t * fps)
        if 0 <= f < n_frames and out[f] is None:
            q = Pm @ np.array([pos[0], pos[1], max(0.0, pos[2]), 1.0])
            if abs(q[2]) > 1e-9:
                out[f] = (float(q[0] / q[2]), float(q[1] / q[2]))
        vel[2] -= G * dt
        pos = pos + vel * dt
        if not rolling and pos[2] <= 0 and vel[2] < 0:
            pos[2] = 0.0
            if bounces == 0:
                bounce_out = PITCH_LEN - float(pos[1])
            vel[2] = -RESTITUTION * vel[2]
            sp = float(np.hypot(vel[0], vel[1])) * FRICTION
            head = np.arctan2(vel[0], vel[1])
            vel[0], vel[1] = sp * np.sin(head), sp * np.cos(head)
            bounces += 1
            if vel[2] < 0.35:
                vel[2] = 0.0; rolling = True
        if rolling:
            pos[2] = 0.0; vel[2] = 0.0
        if cross is None and pos[1] >= PITCH_LEN:
            cross = (float(pos[0]), float(pos[2]))
        t += dt
    return out, dict(bounce_out=bounce_out, cross=cross, bounces=bounces)


def consistent(r, out, pts, info):
    """NOT USED - kept as a record of a guard that was written and then rejected.

    The case: on David's nets clip the arc ends without a bounce while detections carry on for
    another ten frames, and the obvious reading was a WRONG SCALE - two stumps that are not 20.12 m
    apart, so the ball "crosses the stumps" long before the tracker watches it come down.

    It does not survive scrutiny. The camera pose is sane (5.1 m back, 1.83 m up, horizon 144 px
    above the far stumps - all inside the eval set's range), the fit is well determined (rms has a
    clear minimum, 1.89 px at 108 km/h, rising to 2.5-4.5 either side), and the same pixels are
    equally well explained by a real delivery: a 108 km/h FULL TOSS that passed the batsman
    waist-high and landed behind them. Every detection after the crossing is then the ball still
    falling, which is exactly what the image shows.

    So this guard would reject a legitimate delivery on the strength of a hypothesis that the data
    does not support. Whether that clip is a full toss is a question about the clip, answerable by
    watching it - not by a threshold. Left here so the next person does not re-derive it.
    """
    if info.get("bounce_out") is not None:
        return True
    drawn = [i for i, q in enumerate(out) if q is not None]
    if not drawn:
        return False
    after = [f for f in pts if f > drawn[-1] + 1]
    return len(after) < 3
