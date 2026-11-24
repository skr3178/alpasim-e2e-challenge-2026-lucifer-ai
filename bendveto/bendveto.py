"""Bend veto for lf-009: port of our October off-road veto
from the old (8, 3) plan at 0.5 s spacing to lf-009's (64, 3) plan at 0.1 s spacing. Pure numpy, model outputs only.

Rule (parameters copied unchanged from October; NOT re-tuned on lf-009 scenes):
  * acts only from the second planning call of a session, and only on a bend: plan heading change >= 10 deg between
    the 1.0 s and 2.5 s poses;
  * flags the plan when the predicted road surface P[road] + P[centerline] + P[vehicle] under the evaluator footprint
    (5.176 x 2.297 m, centre 1.461 m ahead of the rear axle, corners inflated 0.35 m) drops below 0.5 at the
    1.0 / 1.5 / 2.0 / 2.5 s poses;
  * then tries lateral ramps plan(t) + d * s(t) * n(t), d in (+0.5, -0.5, +1.0, -1.0) m, s(t) = clip((t - 0.5) / 1.5, 0, 1),
    with a heading correction consistent with the offset (candidate dropped if the correction exceeds 12 deg);
    candidates touching a confident vehicle box (class 0, score >= 0.3, inflated 0.5 m) are dropped;
  * takes the smallest |d| whose road score is >= 0.7, else the best candidate if it beats the base by 0.25,
    otherwise keeps the model's plan.
Port changes only: pose indices for the 0.1 s grid; the heading correction is evaluated at every one of the 64 poses."""
import math

import numpy as np

PPM, MIN_X, MIN_Y = 4.0, 0.0, -32.0          # BEV raster: col = (x - MIN_X) * PPM, row = (y - MIN_Y) * PPM
EGO_L, EGO_W, RIG_TO_CENTRE = 5.176, 2.297, 1.461
DT = 0.1
BAND = (9, 14, 19, 24)                       # 1.0, 1.5, 2.0, 2.5 s: the poses the October rule scored
THRESHOLD, ACCEPT, GAIN = 0.5, 0.7, 0.25
MARGIN = 0.35
BEND_DEG = 10.0
DELTAS = (0.5, -0.5, 1.0, -1.0)
BOX_INFLATE = 0.5
MAX_CORRECTION_DEG = 12.0
VEHICLE_CLASS, MIN_SCORE = 0, 0.3


def wrap(a):
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def bilinear(prob, x, y):
    """Bilinear lookup of a (rows=y, cols=x) raster at rig metres; outside the raster -> NaN (unknown)."""
    h, w = prob.shape
    cx = (np.asarray(x, float) - MIN_X) * PPM - 0.5
    cy = (np.asarray(y, float) - MIN_Y) * PPM - 0.5
    c0, r0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
    fx, fy = cx - c0, cy - r0
    out = np.full(np.shape(cx), np.nan)
    ok = (c0 >= 0) & (c0 < w - 1) & (r0 >= 0) & (r0 < h - 1)
    c, r, a, b = c0[ok], r0[ok], fx[ok], fy[ok]
    out[ok] = (prob[r, c] * (1 - a) * (1 - b) + prob[r, c + 1] * a * (1 - b)
               + prob[r + 1, c] * (1 - a) * b + prob[r + 1, c + 1] * a * b)
    return out


def footprint(x, y, yaw, margin):
    """Corners inflated by ``margin`` plus a 5 x 3 interior grid of the evaluator footprint at a rig pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = x + RIG_TO_CENTRE * c, y + RIG_TO_CENTRE * s
    hl, hw = EGO_L / 2 + margin, EGO_W / 2 + margin
    pts = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    pts += [(u, v) for u in np.linspace(-EGO_L / 2, EGO_L / 2, 5) for v in np.linspace(-EGO_W / 2, EGO_W / 2, 3)]
    a = np.asarray(pts, float)
    return np.stack([cx + c * a[:, 0] - s * a[:, 1], cy + s * a[:, 0] + c * a[:, 1]], axis=1)


def road_score(plan, p_road, margin=MARGIN):
    """Minimum predicted road-surface probability over the footprints of the band poses (unknown samples ignored)."""
    vals = []
    for i in BAND:
        pts = footprint(plan[i, 0], plan[i, 1], plan[i, 2], margin)
        v = bilinear(p_road, pts[:, 0], pts[:, 1])
        v = v[np.isfinite(v)]
        vals.append(float(v.min()) if len(v) else 1.0)
    return min(vals)


def touches_box(plan, vehicles):
    if not len(vehicles):
        return False
    pts = np.vstack([footprint(plan[i, 0], plan[i, 1], plan[i, 2], 0.0) for i in BAND])
    for box in vehicles:
        c, s = math.cos(float(box[4])), math.sin(float(box[4]))
        d = pts - box[:2]
        u, v = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        if np.any((np.abs(u) <= box[2] / 2 + BOX_INFLATE) & (np.abs(v) <= box[3] / 2 + BOX_INFLATE)):
            return True
    return False


def lateral_candidate(plan, delta):
    """plan(t) + delta * s(t) * n(t) with a consistent heading correction; None if the correction exceeds 12 deg."""
    n = len(plan)
    times = DT * np.arange(1, n + 1)
    ramp = np.clip((times - 0.5) / 1.5, 0.0, 1.0)
    out = np.array(plan, dtype=np.float64, copy=True)
    out[:, 0] += delta * ramp * -np.sin(plan[:, 2])
    out[:, 1] += delta * ramp * np.cos(plan[:, 2])
    t_all = np.concatenate([[0.0], times])
    e_all = np.concatenate([[0.0], delta * ramp])
    xy_all = np.vstack([[0.0, 0.0], np.asarray(plan, float)[:, :2]])
    for i in range(n):
        j = i + 1
        lo, hi = j - 1, min(j + 1, n)
        de_dt = (e_all[hi] - e_all[lo]) / (t_all[hi] - t_all[lo])
        speed = float(np.linalg.norm(xy_all[hi] - xy_all[lo]) / (t_all[hi] - t_all[lo]))
        corr = math.atan2(de_dt, max(speed, 0.5))
        if abs(math.degrees(corr)) > MAX_CORRECTION_DEG:
            return None
        out[i, 2] = float(wrap(plan[i, 2] + corr))
    return out


def bend_veto(plan, p_road, boxes, call_index):
    """plan (64, 3) rig frame; p_road (256, 256) surface probability; boxes (K, 8) x,y,w,h,yaw,v,class,score.
    Returns (plan_out, info). info['applied'] is True only when the plan was replaced."""
    plan = np.asarray(plan, dtype=np.float64)
    base = road_score(plan, p_road)
    bend = abs(math.degrees(float(wrap(plan[BAND[-1], 2] - plan[BAND[0], 2]))))
    info = dict(applied=False, base=round(base, 3), bend_deg=round(bend, 1), flagged=False, delta=0.0, call=int(call_index))
    if call_index < 1 or base >= THRESHOLD or bend < BEND_DEG:
        return plan, info
    info['flagged'] = True
    b = np.asarray(boxes, float).reshape(-1, 8) if boxes is not None else np.zeros((0, 8))
    vehicles = b[(b[:, 6] == VEHICLE_CLASS) & (b[:, 7] >= MIN_SCORE)] if len(b) else b
    scored = []
    for d in DELTAS:
        cand = lateral_candidate(plan, d)
        if cand is None or touches_box(cand, vehicles):
            continue
        scored.append((d, road_score(cand, p_road), cand))
    scored.sort(key=lambda it: (abs(it[0]), -it[1]))
    good = [it for it in scored if it[1] >= ACCEPT]
    chosen = good[0] if good else max(scored, key=lambda it: it[1], default=None)
    info['candidates'] = len(scored)
    if chosen is None or (not good and chosen[1] < base + GAIN):
        info['reason'] = 'no_candidate'
        return plan, info
    info.update(applied=True, delta=chosen[0], chosen=round(chosen[1], 3), reason='shifted')
    return chosen[2], info
