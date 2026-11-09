import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# On Apple GPUs, let ops MPS doesn't support fall back to CPU. Must be set before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

# post_selection.py lives in the repo root next to utils.py; make it importable from this subfolder.
sys.path.insert(0, str(HERE.parent))
from post_selection import (  # noqa: E402
    DEFAULTS, MAX_PAIR_CANDS, PAIR_TERMS, PAIR_WEIGHTS_DEFAULT, TOP_PAIRS_LOGGED,
    Cand, assign_clusters, choose_by_cluster, containment, is_post_like, make_cand,
    pair_components, pair_weights, rank_pairs,
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

POST_PROMPT = "vertical pole"
BAR_PROMPT = "horizontal pole"

# ---- method 3 (crossbar) ----
STUB_FRAC = 0.15        # bar centreline may sit this fraction of a post's height ABOVE its mask top
BAR_UPPER_FRAC = 0.60   # ...and up to this fraction BELOW it (a reflection-inflated mask has its top too high)
HANG_NEAR_FRAC = 0.25   # a hanging post's top may be this fraction of its height below the bar
HANG_X_MARGIN = 0.10    # post top must be within the bar's x-extent +- this fraction of the span
BAR_TIE = 0.05          # bars scoring within this of the best are tied; the LOWER one wins (real, not reflected)

# Everything above that is not a CLI flag, logged with each run so it can be reproduced.
CONSTANTS = {"PAIR_TERMS": list(PAIR_TERMS), "MAX_PAIR_CANDS": MAX_PAIR_CANDS, "STUB_FRAC": STUB_FRAC,
             "BAR_UPPER_FRAC": BAR_UPPER_FRAC, "HANG_NEAR_FRAC": HANG_NEAR_FRAC,
             "HANG_X_MARGIN": HANG_X_MARGIN, "BAR_TIE": BAR_TIE}

# ---- drawing ----
ROLE_COLORS = {
    "POST-L": (60, 220, 60),
    "POST-R": (255, 140, 0),
    "POST-?": (60, 220, 60),
    "CROSSBAR": (255, 235, 0),
}
DROPPED_COLOR = (255, 70, 70)
OTHER_COLOR = (150, 150, 150)
BAR_CAND_COLOR = (120, 200, 255)
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (170, 110, 40),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def find_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:            # very old Pillow
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Model (torch / transformers are imported lazily so the rest of the file stays importable)
# ---------------------------------------------------------------------------

def pick_device(requested=""):
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device):
    from transformers import Sam3Model, Sam3Processor
    print("Loading facebook/sam3 (needs `hf auth login` the first time) ...")
    start = time.time()
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    print(f"Model loaded in {time.time() - start:.1f}s on {device}\n")
    return model, processor


def segment(image, prompt, model, processor, device, threshold, mask_threshold):
    """
    One SAM3 text-prompt call, same call pattern as annotate_frame.py.

    Returns numpy arrays sorted by score, highest first:
        masks  [N, H, W] uint8 (0/1),  boxes [N, 4] float xyxy pixels,  scores [N] float
    """
    import torch

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    w, h = image.size
    if len(results["masks"]) == 0:
        return np.zeros((0, h, w), np.uint8), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    masks = results["masks"].detach().cpu().numpy().astype(np.uint8)
    boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = results["scores"].detach().cpu().numpy().astype(np.float32)

    order = np.argsort(-scores, kind="stable")
    return masks[order], boxes[order], scores[order]


# ---------------------------------------------------------------------------
# Candidates: one SAM3 instance plus the geometry the selection methods need
# ---------------------------------------------------------------------------

def gather(image, prompt, kind, model, processor, device, args):
    masks, _boxes, scores = segment(image, prompt, model, processor, device,
                                    args.threshold, args.mask_threshold)
    cands = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        c = make_cand(f"{kind}{i}", kind, s, m)
        if c is not None:
            cands.append(c)
    return cands


# ---- candidate cache (so thresholds can be re-tuned without re-running SAM3) ----

def save_candidates(path, vs, hs, shape, args):
    def pack(cs):
        if not cs:
            return np.zeros((0, 0), np.uint8)
        return np.packbits(np.stack([c.mask.reshape(-1) for c in cs]), axis=1)

    meta = {"shape": list(shape), "prompt": args.prompt, "crossbar_prompt": args.crossbar_prompt,
            "threshold": args.threshold, "mask_threshold": args.mask_threshold}
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        v_masks=pack(vs), v_scores=np.array([c.score for c in vs], np.float32),
        h_masks=pack(hs), h_scores=np.array([c.score for c in hs], np.float32),
    )


def load_candidates(path, shape, args):
    """Return (vs, hs) from the cache, or None if it is missing or was made with other settings."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            want = {"shape": list(shape), "prompt": args.prompt, "crossbar_prompt": args.crossbar_prompt,
                    "threshold": args.threshold, "mask_threshold": args.mask_threshold}
            if meta != want:
                print(f"  cache {path.name} was made with different settings -> re-running SAM3")
                return None
            h, w = shape

            def unpack(masks, scores, kind):
                out = []
                for i, s in enumerate(scores):
                    m = np.unpackbits(masks[i], count=h * w).reshape(h, w).astype(bool)
                    c = make_cand(f"{kind}{i}", kind, s, m)
                    if c is not None:
                        out.append(c)
                return out

            return (unpack(z["v_masks"], z["v_scores"], "V"),
                    unpack(z["h_masks"], z["h_scores"], "H"))
    except Exception as exc:                       # corrupt / older cache: just recompute
        print(f"  could not read cache {path.name} ({exc}) -> re-running SAM3")
        return None


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def method_cluster_score(pool, n_clusters, W, args):
    posts, winners, groups = choose_by_cluster(pool, W, args.min_post_sep)
    per_cluster = []
    for cid in sorted(groups):
        g = sorted(groups[cid], key=lambda c: -c.score)
        win = winners[cid]
        others = " ".join(f"{c.uid} {c.score:.2f}" for c in g if c is not win)
        per_cluster.append(f"c{cid}: keep {win.uid} {win.score:.2f}" + (f" (over {others})" if others else ""))
    return {
        "posts": posts,
        "details": [
            "How it works: group candidates whose masks overlap (containment >= "
            f"{args.containment:g}, centres within {args.same_post_dx:.0%} of width), keep the highest "
            f"SAM score in each group, then take the best 2 groups at least {args.min_post_sep:.0%} of width apart.",
            f"{len(pool)} post-like candidates in {len(groups)} groups ({n_clusters} before the post-like filter). "
            + "; ".join(per_cluster),
        ],
    }


def method_pairwise(pool, W, H, args):
    scored = rank_pairs(pool, W, H, args)
    weights = pair_weights(args)
    n_pool = min(len(pool), MAX_PAIR_CANDS)

    top_pairs = [{"pair": [l.uid, r.uid], "score": round(t, 4), "sep_px": round(sep, 1),
                  "components": {k: round(v, 3) for k, v in comps.items()}}
                 for t, comps, l, r, sep in scored[:TOP_PAIRS_LOGGED]]

    names = {"conf": "confidence", "len": "length match", "level": "posts perpendicular to joining line",
             "par": "top/bottom lines parallel", "size": "separation vs widest pair",
             "height": "shorter-post height vs tallest pair"}
    wtxt = " + ".join(f"{weights[k]:.2f} {names[k]}" for k in PAIR_TERMS)
    skips = ["they overlap", f"the posts are < {args.min_post_sep:.0%} of width apart"]
    if args.max_pair_aspect > 0:
        skips.append(f"separation > {args.max_pair_aspect:g} x the shorter post's height (unless a post is cut by the frame)")
    details = [
        f"How it works: score every pair of post-like candidates = {wtxt}. "
        f"A pair is skipped if {'; or '.join(skips)}. Best pair wins.",
    ]
    if not scored:
        # Same fallback as post_selection.select_posts_pairwise: the single best post, if any.
        single, _, _ = choose_by_cluster(pool, W, args.min_post_sep)
        single = sorted(single, key=lambda c: -c.score)[:1]
        details.append(f"{n_pool} post-like candidates -> 0 valid pairs. "
                       + (f"FALLBACK to the single best post {single[0].uid} {single[0].score:.2f}." if single
                          else "Nothing selected."))
        return {"posts": single, "details": details, "score": None, "top_pairs": []}

    def fmt(rank, t, comps, l, r):
        return (f"{rank}) {l.uid}+{r.uid} {t:.2f} ("
                + ", ".join(f"{k} {comps[k]:.2f}" for k in PAIR_TERMS) + ")")

    details.append(f"{n_pool} post-like candidates -> {len(scored)} valid pairs. "
                   + "   ".join(fmt(i + 1, t, comps, l, r) for i, (t, comps, l, r, _) in enumerate(scored[:3])))
    total, comps, left, right, _ = scored[0]
    return {"posts": [left, right], "details": details, "score": total, "components": comps,
            "top_pairs": top_pairs}


def fit_centerline(mask):
    """Straight-line fit y = m*x + b through the per-column mean y of a (thin, roughly horizontal) mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    xmin = xs.min()
    counts = np.bincount(xs - xmin)
    sums = np.bincount(xs - xmin, weights=ys)
    ok = counts > 0
    if ok.sum() < 3:
        return None
    cols = np.nonzero(ok)[0] + xmin
    m, b = np.polyfit(cols, sums[ok] / counts[ok], 1)
    return float(m), float(b)


def find_crossbar(hs, seeds, W, H, args):
    """
    Best horizontal candidate that spans the two seed posts. Uses only the seeds' x positions and
    rough tops, so it still works when the seed masks are noisy.
    Returns ((cand, m, b), notes) or (None, notes).
    """
    if len(seeds) < 2:
        return None, f"need 2 seed posts, got {len(seeds)}"
    L, R = seeds
    xl, xr = L.cx, R.cx
    span = max(xr - xl, 1.0)
    tol = args.below_tol * H

    keep, why = [], Counter()
    for c in hs:
        if c.elong < args.min_elong:
            why["not thin"] += 1
            continue
        fit = fit_centerline(c.mask)
        if fit is None:
            why["too short to fit"] += 1
            continue
        m, b = fit
        if math.degrees(math.atan(abs(m))) > args.max_bar_tilt:
            why["too tilted"] += 1
            continue
        coverage = max(0.0, min(c.x1, xr) - max(c.x0, xl)) / span
        if coverage < args.min_coverage:
            why["does not span the posts"] += 1
            continue
        if (xl - c.x0) > args.bar_overhang * span or (c.x1 - xr) > args.bar_overhang * span:
            why["sticks out past the posts"] += 1
            continue
        if not all(s.y0 - STUB_FRAC * s.h - tol <= m * s.cx + b <= s.y0 + BAR_UPPER_FRAC * s.h for s in (L, R)):
            why["not at the post tops"] += 1
            continue
        keep.append((c, m, b))

    if not keep:
        reasons = ", ".join(f"{k}: {v}" for k, v in why.items()) or "no horizontal candidates"
        return None, f"no crossbar among {len(hs)} horizontal candidates ({reasons})"
    best = max(c.score for c, _, _ in keep)
    tied = [t for t in keep if t[0].score >= best - BAR_TIE]
    choice = max(tied, key=lambda t: t[0].cy)          # lowest in the image = the real bar, not its reflection
    return choice, f"{len(keep)} of {len(hs)} horizontal candidates pass the filters"


def method_crossbar(pool, hs, seeds, seed_name, W, H, args):
    details = [
        f"How it works: take the 2 seed posts (from {seed_name}), find a crossbar among '{args.crossbar_prompt}' "
        f"candidates that spans them (coverage >= {args.min_coverage:.0%}, tilt <= {args.max_bar_tilt:g} deg, "
        "sits at the post tops), then keep only post candidates whose TOP hangs just below the bar and "
        "re-pick with method 1's rule. A reflected post pokes above the bar and is dropped.",
    ]
    bar, note = find_crossbar(hs, seeds, W, H, args)
    if bar is None:
        details.append("FAILED: " + note + ". Seed posts are shown in red, nothing is selected.")
        return {"posts": [], "details": details, "bar": None, "dropped": [(s, "no crossbar found") for s in seeds]}

    c, m, b = bar
    tol = args.below_tol * H
    span = max(c.x1 - c.x0, 1.0)
    hanging, rejected = [], []
    for v in pool:
        tx, ty = v.top
        if not (c.x0 - HANG_X_MARGIN * span <= tx <= c.x1 + HANG_X_MARGIN * span):
            rejected.append((v, "outside bar x-range"))
            continue
        d = ty - (m * tx + b)                           # > 0: the post top is below the bar line
        if d < -tol:
            rejected.append((v, "top above bar"))
        elif d > max(tol, HANG_NEAR_FRAC * v.h):
            rejected.append((v, "top far below bar"))
        else:
            hanging.append(v)

    posts, _, _ = choose_by_cluster(hanging, W, args.min_post_sep)
    dropped = [(s, dict((id(v), r) for v, r in rejected).get(id(s), "lost the re-pick"))
               for s in seeds if s not in posts]
    details.append(f"Crossbar {c.uid} {c.score:.2f} ({note}). {len(hanging)} of {len(pool)} post-like "
                   f"candidates hang from it; re-picked {len(posts)}."
                   + (" Seed posts dropped: " + ", ".join(f"{s.uid} ({r})" for s, r in dropped) if dropped else
                      " Both seed posts were kept."))
    return {"posts": posts, "details": details, "bar": bar, "dropped": dropped}


def run_methods(vs, hs, W, H, args):
    """Apply the three selection ideas to one image's candidates. Pure function: no model, no drawing."""
    for c in vs:
        c.post_like = is_post_like(c, H, args)
    n_clusters = assign_clusters(vs, W, args)
    pool = [c for c in vs if c.post_like]

    m1 = method_cluster_score(pool, n_clusters, W, args)
    m2 = method_pairwise(pool, W, H, args)
    if args.crossbar_seed == "cluster":
        seeds, seed_name = m1["posts"], "method 1 (cluster + score)"
    else:
        seeds, seed_name = m2["posts"], "method 2 (pairwise)"
    m3 = method_crossbar(pool, hs, seeds, seed_name, W, H, args)
    return m1, m2, m3, n_clusters


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

@dataclass
class Item:
    cand: Cand
    color: tuple
    label: str
    fill: float = 0.5        # 0 = outline only


def dilate(m, n):
    out = m.copy()
    for _ in range(n):
        d = out.copy()
        d[1:] |= out[:-1]
        d[:-1] |= out[1:]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def edge_of(m):
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def wrap_text(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def render(image, title, details, items, legend, segments=()):
    """
    Frame + labelled masks under a banner. `items` are drawn gray-outline-first so selected
    (filled) masks sit on top. `segments` are extra ((x0,y0),(x1,y1),color) lines.
    """
    image = image.convert("RGB")
    w, h = image.size
    fsize = max(14, w // 60)
    font = load_font(fsize)
    lw = max(2, w // 500)

    arr = np.asarray(image).astype(np.float32).copy()
    ordered = sorted(items, key=lambda it: it.fill)            # outlines first, filled last
    for it in ordered:
        if it.fill > 0:
            arr[it.cand.mask] = arr[it.cand.mask] * (1 - it.fill) + np.array(it.color, np.float32) * it.fill
    for it in ordered:
        arr[dilate(edge_of(it.cand.mask), lw - 1)] = it.color
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)

    for p0, p1, color in segments:
        draw.line([p0, p1], fill=color, width=lw * 2)

    placed = []
    for it in ordered:
        c = it.cand
        draw.rectangle([c.x0, c.y0, c.x1, c.y1], outline=it.color, width=1 if it.fill == 0 else lw)
        l, t, r, b = draw.textbbox((0, 0), it.label, font=font)
        tw, th = r - l, b - t
        x = int(min(max(c.x0, 0), max(0, w - tw - 8)))
        y = c.y0 - th - 8 if c.y0 - th - 8 >= 0 else c.y0 + 2
        for _ in range(14):                                     # nudge down until it clears earlier tags
            rect = [x, y, x + tw + 8, y + th + 6]
            if any(not (rect[2] <= p[0] or p[2] <= rect[0] or rect[3] <= p[1] or p[3] <= rect[1]) for p in placed):
                y += th + 6
            else:
                break
        placed.append(rect)
        draw.rectangle(rect, fill=(0, 0, 0))
        draw.text((x + 4, y + 2 - t), it.label, fill=it.color, font=font)

    # ---- banner ----
    meas = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    pad, line_h = 8, fsize + 6
    max_w = w - 2 * pad
    title_lines = wrap_text(meas, title, font, max_w)
    detail_lines = [ln for d in details for ln in wrap_text(meas, d, font, max_w)]

    legend_rows, row, x = [], [], 0
    for label, color in legend:
        need = fsize + 6 + int(meas.textlength(label, font=font)) + 24
        if row and x + need > max_w:
            legend_rows.append(row)
            row, x = [], 0
        row.append((label, color, x))
        x += need
    if row:
        legend_rows.append(row)

    banner_h = pad * 2 + line_h * (len(title_lines) + len(detail_lines) + len(legend_rows))
    out = Image.new("RGB", (w, h + banner_h), (25, 25, 25))
    out.paste(canvas, (0, banner_h))
    d = ImageDraw.Draw(out)
    y = pad
    for ln in title_lines:
        d.text((pad, y), ln, fill=(255, 255, 255), font=font)
        y += line_h
    for ln in detail_lines:
        d.text((pad, y), ln, fill=(185, 185, 185), font=font)
        y += line_h
    for rowitems in legend_rows:
        for label, color, lx in rowitems:
            d.rectangle([pad + lx, y + 3, pad + lx + fsize, y + 3 + fsize], fill=color)
            d.text((pad + lx + fsize + 6, y), label, fill=(255, 255, 255), font=font)
        y += line_h
    return out


def role_labels(posts):
    return ["POST-L", "POST-R"] if len(posts) == 2 else ["POST-?"] * len(posts)


def method_items(vs, hs, result, show_h):
    """Selected candidates get their role colour; everything else is a thin gray outline."""
    items, chosen = [], {}
    for role, c in zip(role_labels(result["posts"]), result["posts"]):
        chosen[id(c)] = (role, ROLE_COLORS[role])
    bar = result.get("bar")
    if bar is not None:
        chosen[id(bar[0])] = ("CROSSBAR", ROLE_COLORS["CROSSBAR"])
    dropped = {id(c): why for c, why in result.get("dropped", [])}

    for c in list(vs) + (list(hs) if show_h else []):
        if id(c) in chosen:
            role, color = chosen[id(c)]
            items.append(Item(c, color, f"{role} {c.uid} {c.score:.2f}", 0.5))
        elif id(c) in dropped:
            items.append(Item(c, DROPPED_COLOR, f"DROPPED {c.uid} {c.score:.2f} ({dropped[id(c)]})", 0.0))
        else:
            items.append(Item(c, OTHER_COLOR, f"{c.uid} {c.score:.2f}", 0.0))
    return items


def bar_segment(bar):
    c, m, b = bar
    return ((c.x0, m * c.x0 + b), (c.x1, m * c.x1 + b), ROLE_COLORS["CROSSBAR"])


def role_legend(result, extra_other=True):
    legend = [(r, ROLE_COLORS[r]) for r in dict.fromkeys(role_labels(result["posts"]))]
    if result.get("bar") is not None:
        legend.append(("CROSSBAR", ROLE_COLORS["CROSSBAR"]))
    if result.get("dropped"):
        legend.append(("DROPPED seed post", DROPPED_COLOR))
    if extra_other:
        legend.append(("other candidate", OTHER_COLOR))
    return legend or [("nothing selected", OTHER_COLOR)]


def result_summary(result):
    out = {role: c.uid for role, c in zip(role_labels(result["posts"]), result["posts"])}
    if result.get("bar") is not None:
        out["CROSSBAR"] = result["bar"][0].uid
    if result.get("score") is not None:
        out["pair_score"] = round(float(result["score"]), 4)
    if result.get("dropped"):
        out["dropped_seeds"] = {c.uid: why for c, why in result["dropped"]}
    return out


def make_grid(images, cols=2, max_w=960):
    ims = []
    for im in images:
        s = min(1.0, max_w / im.width)
        ims.append(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS))
    cw, ch = max(i.width for i in ims), max(i.height for i in ims)
    rows = (len(ims) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cw, rows * ch), (25, 25, 25))
    for k, im in enumerate(ims):
        grid.paste(im, ((k % cols) * cw, (k // cols) * ch))
    return grid


def render_all_methods(image, vs, hs, m1, m2, m3, n_clusters, args):
    """The five images for one frame, in file order."""
    W, H = image.size

    # 00: every candidate, nothing chosen
    items = []
    for c in vs:
        color = PALETTE[c.cluster % len(PALETTE)]
        items.append(Item(c, color, f"{c.uid} c{c.cluster} {c.score:.2f}" + ("" if c.post_like else " !"), 0.35))
    for c in hs:
        items.append(Item(c, BAR_CAND_COLOR, f"{c.uid} {c.score:.2f}", 0.0))
    n_pl = sum(c.post_like for c in vs)
    img00 = render(
        image,
        "ALL CANDIDATES (nothing selected)",
        [f'V = "{args.prompt}" instances ({len(vs)}), coloured by overlap cluster (c#); '
         f"{n_pl} pass the post-like filter, '!' marks the ones that do not. "
         f'H = "{args.crossbar_prompt}" instances ({len(hs)}), blue outlines. Number after the id = SAM score.'],
        items,
        [(f"cluster c{k}", PALETTE[k % len(PALETTE)]) for k in range(min(n_clusters, 6))]
        + [("H candidate", BAR_CAND_COLOR)],
    )

    img01 = render(image, "METHOD 1 - CLUSTER + SCORE", m1["details"],
                   method_items(vs, hs, m1, False), role_legend(m1))
    img02 = render(image, "METHOD 2 - PAIRWISE", m2["details"],
                   method_items(vs, hs, m2, False), role_legend(m2))
    segs = [bar_segment(m3["bar"])] if m3["bar"] is not None else []
    img03 = render(image, "METHOD 3 - CROSSBAR RESELECT", m3["details"],
                   method_items(vs, hs, m3, True), role_legend(m3), segs)
    return [img00, img01, img02, img03]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_NAMES = ["00_all_candidates.png", "01_cluster_score.png", "02_pairwise.png", "03_crossbar_reselect.png"]


def build_parser():
    ap = argparse.ArgumentParser(description="Compare ways of selecting clean gate posts / crossbar from SAM3 output.")
    ap.add_argument("--images_dir", default=str(HERE / "images"), help="Folder of test frames")
    ap.add_argument("--out_dir", default=str(HERE / "visualization"), help="Where overlays are written")
    ap.add_argument("--prompt", default=POST_PROMPT, help="Prompt for the posts")
    ap.add_argument("--crossbar_prompt", default=BAR_PROMPT, help="Prompt used only to find the crossbar (method 3)")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="SAM3 instance confidence threshold (default 0.3; the real pipeline uses 0.5)")
    ap.add_argument("--mask_threshold", type=float, default=0.5, help="Mask binarisation threshold")
    ap.add_argument("--device", default="", help="Override device (cuda:0, mps, cpu). Default: cuda > mps > cpu")
    ap.add_argument("--reuse_candidates", action="store_true",
                    help="Load candidates.npz instead of running SAM3 (only if made with the same prompts/thresholds)")
    ap.add_argument("--no_save_candidates", action="store_true", help="Do not write candidates.npz")

    g = ap.add_argument_group("clustering / top-2 (fractions of image width unless noted)")
    g.add_argument("--containment", type=float, default=DEFAULTS.containment,
                   help="Masks overlapping at least this fraction of the smaller one are the same post")
    g.add_argument("--same_post_dx", type=float, default=DEFAULTS.same_post_dx,
                   help="...and only if their centres are within this fraction of image width")
    g.add_argument("--min_post_sep", type=float, default=DEFAULTS.min_post_sep,
                   help="The two chosen posts must be at least this far apart horizontally (lower for distant gates)")

    g = ap.add_argument_group("post-like filter")
    g.add_argument("--min_elong", type=float, default=DEFAULTS.min_elong, help="Min length/width ratio of the mask's principal axes")
    g.add_argument("--max_tilt", type=float, default=DEFAULTS.max_tilt, help="Max degrees off vertical")
    g.add_argument("--min_height", type=float, default=DEFAULTS.min_height, help="Min mask height as a fraction of image height")

    g = ap.add_argument_group("pairwise (method 2)")
    g.add_argument("--pair_weights", nargs=len(PAIR_TERMS), type=float, default=list(PAIR_WEIGHTS_DEFAULT),
                   metavar=tuple(t.upper() for t in PAIR_TERMS),
                   help="Weights for " + " ".join(PAIR_TERMS) + " (normalised to sum 1). Untuned defaults.")
    g.add_argument("--pair_level_tol", type=float, default=DEFAULTS.pair_level_tol,
                   help="Degrees of post-vs-joining-line mismatch at which the level term hits 0")
    g.add_argument("--pair_parallel_tol", type=float, default=DEFAULTS.pair_parallel_tol,
                   help="Degrees between top line and bottom line at which the parallel term hits 0")
    g.add_argument("--pair_max_overlap", type=float, default=DEFAULTS.pair_max_overlap,
                   help="Masks overlapping more than this (fraction of the smaller) are one post, not a pair")
    g.add_argument("--max_pair_aspect", type=float, default=DEFAULTS.max_pair_aspect,
                   help="Skip a pair whose separation exceeds this x the shorter post's height "
                        "(0 = off; ignored if a post is cut by the frame edge). Handbook gate is ~2.4:1")

    g = ap.add_argument_group("crossbar (method 3)")
    g.add_argument("--crossbar_seed", choices=["pairwise", "cluster"], default="pairwise",
                   help="Which method supplies the two seed posts")
    g.add_argument("--max_bar_tilt", type=float, default=30.0, help="Max degrees off horizontal")
    g.add_argument("--min_coverage", type=float, default=0.8, help="Fraction of the post-to-post span the bar must cover")
    g.add_argument("--bar_overhang", type=float, default=0.10,
                   help="How far (fraction of span) the bar may stick out past each post; rejects wall lines")
    g.add_argument("--below_tol", type=float, default=0.015,
                   help="Slack (fraction of image height) when deciding if a post top is above/below the bar")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    images = find_images(args.images_dir)
    if not images:
        print(f"No .png/.jpg/.jpeg images found in {args.images_dir}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{len(images)} image(s); posts: "{args.prompt}", crossbar: "{args.crossbar_prompt}" -> {out_dir}')
    print(f"threshold={args.threshold}  mask_threshold={args.mask_threshold}  "
          f"reuse_candidates={args.reuse_candidates}\n")

    model = processor = device = None
    stem_counts = Counter(p.stem for p in images)
    summary, n_ok, n_fail = {}, 0, 0

    for img_path in images:
        # One folder per image. If two files share a stem (a.png and a.jpg), both get the suffix.
        stem = img_path.stem
        name = stem if stem_counts[stem] == 1 else f"{stem}_{img_path.suffix.lstrip('.').lower()}"
        folder = out_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        print(f"=== {img_path.name} ({W}x{H}) ===")

        try:
            cached = load_candidates(folder / "candidates.npz", (H, W), args) if args.reuse_candidates else None
            if cached is not None:
                vs, hs = cached
                print("  candidates: loaded from cache")
            else:
                if model is None:
                    device = pick_device(args.device)
                    model, processor = load_model(device)
                vs = gather(image, args.prompt, "V", model, processor, device, args)
                bar_ok = True
                try:
                    hs = gather(image, args.crossbar_prompt, "H", model, processor, device, args)
                except Exception:
                    traceback.print_exc()
                    print("  crossbar prompt failed: method 3 will have no crossbar candidates (not cached)")
                    hs, bar_ok = [], False
                if bar_ok and not args.no_save_candidates:   # never cache a failed prompt as "found nothing"
                    save_candidates(folder / "candidates.npz", vs, hs, (H, W), args)

            m1, m2, m3, n_clusters = run_methods(vs, hs, W, H, args)
            images_out = render_all_methods(image, vs, hs, m1, m2, m3, n_clusters, args)
            for fname, im in zip(OUTPUT_NAMES, images_out):
                im.save(folder / fname)
            make_grid(images_out).save(folder / "compare.png")

            methods = {"cluster_score": result_summary(m1), "pairwise": result_summary(m2),
                       "crossbar_reselect": result_summary(m3)}
            record = {
                "image": img_path.name, "size": [W, H],
                "params": {k: v for k, v in vars(args).items() if k not in ("images_dir", "out_dir")},
                "candidates": [
                    {"uid": c.uid, "kind": c.kind, "score": round(c.score, 4), "area_px": c.area,
                     "box_xyxy": [c.x0, c.y0, c.x1 + 1, c.y1 + 1], "cluster": c.cluster if c.kind == "V" else None,
                     "post_like": c.post_like if c.kind == "V" else None,
                     "tilt_deg": round(c.tilt, 1), "elongation": round(c.elong, 1)}
                    for c in list(vs) + list(hs)],
                "methods": methods,
                "pairwise_top_pairs": m2.get("top_pairs", []),
                "constants": CONSTANTS,
            }
            (folder / "selections.json").write_text(json.dumps(record, indent=2))
            summary[name] = methods
        except Exception:
            traceback.print_exc()
            print(f"  FAILED on {img_path.name} (continuing)\n")
            n_fail += 1
            continue

        print(f"  {len(vs)} vertical, {len(hs)} horizontal candidates; {n_clusters} overlap clusters")
        for c in vs:
            print(f"    {c.uid}: score {c.score:.2f}  cluster c{c.cluster}  h={c.h}px  tilt={c.tilt:.0f}  "
                  f"elong={c.elong:.0f}  {'' if c.post_like else '(not post-like)'}")
        for label, res in methods.items():
            print(f"  {label:18s} {res}")
        for k, tp in enumerate(m2.get("top_pairs", [])):
            print(f"    pair #{k + 1}: {'+'.join(tp['pair'])}  {tp['score']:.3f}  sep={tp['sep_px']:.0f}px  {tp['components']}")
        pick = {k: frozenset(v for r, v in m.items() if r in ("POST-L", "POST-R", "POST-?")) for k, m in methods.items()}
        if not any(pick.values()):
            verdict = "NO methods selected any post"
        elif len(set(pick.values())) == 1:
            verdict = "AGREE across all 3 methods"
        else:
            verdict = "DIFFER between methods"
        print(f"  posts {verdict}\n")
        n_ok += 1

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done: {n_ok} image(s) processed, {n_fail} failed. Open {out_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())