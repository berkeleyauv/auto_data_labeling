import math
from dataclasses import dataclass, field
from itertools import combinations
from types import SimpleNamespace
 
import numpy as np
 
# ---- pairwise score terms (order of `pair_weights`) ----
PAIR_TERMS = ("conf", "len", "level", "par", "size")
PAIR_WEIGHTS_DEFAULT = (0.30, 0.15, 0.20, 0.15, 0.20)     # normalised to sum 1 at run time
MAX_PAIR_CANDS = 30            # safety cap on n for the n^2 loop
TOP_PAIRS_LOGGED = 5
 
DEFAULTS = SimpleNamespace(
    # clustering (fractions of image width unless noted)
    containment=0.5,           # masks overlapping >= this fraction of the smaller one are the same post
    same_post_dx=0.04,         # ...and only if their centres are within this fraction of image width
    min_post_sep=0.10,         # the two chosen posts must be at least this far apart horizontally
    # post-like filter
    min_elong=4.0,             # min length/width ratio of the mask's principal axes
    max_tilt=30.0,             # max degrees off vertical
    min_height=0.10,           # min mask height as a fraction of image height
    # pairwise
    pair_weights=PAIR_WEIGHTS_DEFAULT,
    pair_level_tol=15.0,       # degrees of post-vs-joining-line mismatch at which the level term hits 0
    pair_parallel_tol=20.0,    # degrees between top line and bottom line at which the parallel term hits 0
    pair_max_overlap=0.2,      # masks overlapping more than this are one post, not a pair
    max_pair_aspect=2.6,       # skip pairs wider than this x the shorter post's height (0 = off)
    # area baseline
    area_ratio=0.5,            # second-largest must be at least this fraction of the largest
)
 
# ---------------------------------------------------------------------------
# Candidates: one SAM3 instance plus the geometry the selectors need
# ---------------------------------------------------------------------------
 
@dataclass
class Cand:
    uid: str                    # "V3" = the 4th instance for the post prompt; "H0" = crossbar prompt
    kind: str                   # "V" or "H"
    score: float
    mask: np.ndarray = field(repr=False)   # bool [H, W]
    area: int = 0
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0
    cx: float = 0.0             # bounding-box centre
    cy: float = 0.0
    top: tuple = (0.0, 0.0)     # mean (x, y) of the top 5% of rows
    bot: tuple = (0.0, 0.0)     # mean (x, y) of the bottom 5% of rows
    tilt: float = 0.0           # angle of the mask's long axis from vertical, degrees (0 = upright)
    tilt_s: float = 0.0         # same, signed: atan2(vx, vy) with the axis pointing DOWN the image
    elong: float = 1.0          # sqrt(major/minor variance): ~1 for a blob, large for a thin bar
    cluster: int = -1
    post_like: bool = True
    idx: int = -1               # index of this instance in the `masks` array it came from
 
    @property
    def w(self):
        return self.x1 - self.x0 + 1
 
    @property
    def h(self):
        return self.y1 - self.y0 + 1
 
 
def make_cand(uid, kind, score, mask, idx=-1):
    """Build a Cand from a binary mask. Returns None for an empty mask."""
    mask = np.asarray(mask) > 0
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    band = max(1, int(round(0.05 * h)))
    top_sel = ys <= y0 + band - 1
    bot_sel = ys >= y1 - band + 1
    top = (float(xs[top_sel].mean()), float(ys[top_sel].mean()))
    bot = (float(xs[bot_sel].mean()), float(ys[bot_sel].mean()))
 
    # Principal axis of the pixel cloud (sub-sampled so huge masks stay cheap).
    px, py = xs, ys
    if xs.size > 20000:
        step = xs.size // 20000 + 1
        px, py = xs[::step], ys[::step]
    tilt, tilt_s, elong = 0.0, 0.0, 1.0
    if px.size >= 3:
        pts = np.stack([px, py]).astype(np.float64)
        evals, evecs = np.linalg.eigh(np.cov(pts))          # ascending eigenvalues
        v = evecs[:, 1]                                      # long axis
        if v[1] < 0:                                         # orient the axis down the image
            v = -v
        tilt_s = float(np.degrees(np.arctan2(v[0], v[1])))
        tilt = abs(tilt_s)
        elong = float(min(999.0, np.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9))))
 
    return Cand(uid=uid, kind=kind, score=float(score), mask=mask, area=int(xs.size),
                x0=x0, y0=y0, x1=x1, y1=y1, cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0,
                top=top, bot=bot, tilt=tilt, tilt_s=tilt_s, elong=elong, idx=idx)
 
def containment(a, b):
    """|a ∩ b| / min(|a|, |b|): 1.0 when the smaller mask lies entirely inside the bigger one."""
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    if x0 > x1 or y0 > y1:
        return 0.0
    inter = np.count_nonzero(a.mask[y0:y1 + 1, x0:x1 + 1] & b.mask[y0:y1 + 1, x0:x1 + 1])
    return inter / max(1, min(a.area, b.area))
 
 
def _to_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
 
 
def build_candidates(masks, scores):
    """
    One Cand per non-empty mask (uid "V<i>", idx = position in `masks`).
    Returns (cands, W, H) where W, H are the mask resolution.
    """
    masks, scores = _to_numpy(masks), _to_numpy(scores)
    if masks.ndim != 3 or len(masks) == 0:
        return [], 0, 0
    H, W = masks.shape[1:]
    cands = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        c = make_cand(f"V{i}", "V", s, m, idx=i)
        if c is not None:
            cands.append(c)
    return cands, W, H
 
 
# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------
 
def is_post_like(c, H, params):
    """Cheap sanity filter so a half-gate blob or a stray speck is never ranked as a post."""
    return c.elong >= params.min_elong and c.tilt <= params.max_tilt and c.h >= params.min_height * H
 
 
def assign_clusters(cands, W, params):
    """
    Union-find over candidates: link two when their masks overlap (containment >= params.containment)
    AND their centres are within params.same_post_dx * W. The x condition stops a huge blob that
    swallows both posts from chaining the two posts into one group. Cluster ids run left to right.
    Returns the number of clusters.
    """
    n = len(cands)
    parent = list(range(n))
 
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
 
    for i in range(n):
        for j in range(i + 1, n):
            if abs(cands[i].cx - cands[j].cx) > params.same_post_dx * W:
                continue
            if containment(cands[i], cands[j]) >= params.containment:
                parent[find(j)] = find(i)
 
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda idx: min(cands[k].cx for k in idx))
    for cid, idx in enumerate(ordered):
        for k in idx:
            cands[k].cluster = cid
    return len(ordered)
 
 
def choose_by_cluster(pool, W, min_sep):
    """Best SAM score per cluster, then the best 2 clusters that are >= min_sep*W apart."""
    groups = {}
    for c in pool:
        groups.setdefault(c.cluster, []).append(c)
    winners = {cid: max(g, key=lambda c: (round(c.score, 3), -c.area)) for cid, g in groups.items()}
 
    chosen = []
    for c in sorted(winners.values(), key=lambda c: (-round(c.score, 3), c.area)):
        if all(abs(c.cx - k.cx) >= min_sep * W for k in chosen):
            chosen.append(c)
        if len(chosen) == 2:
            break
    return sorted(chosen, key=lambda c: c.cx), winners, groups
 
 
def pair_components(a, b, W, H, params):
    """
    Terms for the pair (a left of b), or None if the pair is not valid. The `size` term needs the
    other valid pairs, so it is added in rank_pairs. Returns (components, separation_px).
    """
    sep = abs(b.cx - a.cx)
    if sep < params.min_post_sep * W:
        return None
    if containment(a, b) > params.pair_max_overlap:
        return None
    # Spacing cap: the gate is only ~2.4 post-heights wide head-on and foreshortening only shrinks it.
    # Skipped when a post is cut by the top/bottom of the frame (its visible height is too small).
    cut = a.y0 <= 0 or a.y1 >= H - 1 or b.y0 <= 0 or b.y1 >= H - 1
    if params.max_pair_aspect > 0 and not cut and sep > params.max_pair_aspect * min(a.h, b.h):
        return None
 
    conf = (a.score + b.score) / 2.0
    length = min(a.h, b.h) / float(max(a.h, b.h))
    ang_top = math.degrees(math.atan2(b.top[1] - a.top[1], b.top[0] - a.top[0]))
    ang_bot = math.degrees(math.atan2(b.bot[1] - a.bot[1], b.bot[0] - a.bot[0]))
    par = 1.0 - min(1.0, abs(ang_top - ang_bot) / params.pair_parallel_tol)
 
    # In the pair's own frame: a gate rolled by `roll` degrees (right end lower = positive) turns an
    # upright post into one whose signed tilt is -roll. A pair that is merely OFFSET vertically has a
    # joining line that is tilted while the posts are upright, so the mismatch is large.
    roll = (ang_top + ang_bot) / 2.0
    mismatch = (abs(a.tilt_s + roll) + abs(b.tilt_s + roll)) / 2.0
    level = 1.0 - min(1.0, mismatch / params.pair_level_tol)
 
    return {"conf": conf, "len": length, "level": level, "par": par}, sep
 
def pair_weights(params):
    total = float(sum(params.pair_weights))
    if total <= 0:
        raise ValueError("pair_weights must sum to > 0")
    return {k: float(v) / total for k, v in zip(PAIR_TERMS, params.pair_weights)}
 
def rank_pairs(pool, W, H, params):
    """
    All valid pairs from `pool`, best first, as tuples (total, components, left, right, separation_px).
    The `size` component is the pair's separation relative to the widest valid pair in this image.
    """
    pool = sorted(pool, key=lambda c: -c.score)[:MAX_PAIR_CANDS]
    raw = []
    for a, b in combinations(pool, 2):
        left, right = (a, b) if a.cx <= b.cx else (b, a)
        res = pair_components(left, right, W, H, params)
        if res is not None:
            raw.append((left, right, res[0], res[1]))
 
    weights = pair_weights(params)
    max_sep = max((r[3] for r in raw), default=1.0)
    scored = []
    for left, right, comps, sep in raw:
        comps = dict(comps, size=sep / max_sep)          # relative: the widest valid pair gets 1.0
        scored.append((sum(weights[k] * comps[k] for k in PAIR_TERMS), comps, left, right, sep))
    scored.sort(key=lambda t: -t[0])
    return scored
 
def prepare(masks, scores, params):
    """Candidates -> clusters -> post-like pool. Returns (pool, W, H, n_clusters)."""
    cands, W, H = build_candidates(masks, scores)
    for c in cands:
        c.post_like = is_post_like(c, H, params)
    n_clusters = assign_clusters(cands, W, params)
    return [c for c in cands if c.post_like], W, H, n_clusters
 
# ---------------------------------------------------------------------------
# The three selectors
# ---------------------------------------------------------------------------
 
def select_posts_cluster(masks, scores, params=DEFAULTS):
    pool, W, _H, _n = prepare(masks, scores, params)
    posts, _, _ = choose_by_cluster(pool, W, params.min_post_sep)
    return [c.idx for c in posts]
 
 
def select_posts_pairwise(masks, scores, params=DEFAULTS):
    pool, W, H, _n = prepare(masks, scores, params)
    scored = rank_pairs(pool, W, H, params)
    if scored:
        _, _, left, right, _ = scored[0]
        return [left.idx, right.idx]
    # No valid pair (one post in view, or the spacing cap rejected everything): single best post.
    posts, _, _ = choose_by_cluster(pool, W, params.min_post_sep)
    return [c.idx for c in sorted(posts, key=lambda c: -c.score)[:1]]

def select_posts_area(masks, scores, params=DEFAULTS):
    """Previous behaviour: largest contour area (+ the second largest if >= area_ratio of it)."""
    import cv2                                   # lazy: the other selectors do not need it
    masks = _to_numpy(masks).astype(np.uint8)
    areas = []
    for m in masks:
        contours, _ = cv2.findContours(m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas.append(max((cv2.contourArea(c) for c in contours), default=0.0))
 
    order = np.argsort(areas)[::-1].tolist()
    keep = []
    if order and areas[order[0]] > 0:
        keep.append(order[0])
        if len(order) > 1 and (areas[order[1]] / areas[order[0]]) >= params.area_ratio:
            keep.append(order[1])
        elif len(order) > 1:
            print(f"    -> Discarding second post. Ratio: {areas[order[1]] / areas[order[0]]:.2f} < {params.area_ratio}")
    return keep

