import argparse
import os
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# On Apple GPUs, let ops MPS doesn't support fall back to CPU. Must be set before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MASK_ALPHA = 0.45

DEFAULT_PROMPTS = [
    "vertical black pole",     # what annotate_dataset.py uses today
    "vertical pole",           # colour-free: does it catch two-tone posts and the red divider?
    "vertical red pole",       # the centre divider
    "gate post",
    "horizontal black pole",   # crossbar (commented out in annotate_dataset.py)
    "horizontal pole",
]

# Fixed palette so colours are the same on every run (instance #i always gets the same colour).
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "prompt"


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
# Visualization
# ---------------------------------------------------------------------------

def render_overlay(image, masks, boxes, scores, prompt):
    """Frame + coloured masks + boxes + '#i score' tags, under a banner with the prompt."""
    image = image.convert("RGB")
    w, h = image.size
    font = load_font(max(14, w // 55))
    font_px = getattr(font, "size", 14)
    line = max(2, w // 400)

    canvas = image.convert("RGBA")
    for i, mask in enumerate(masks):
        color = PALETTE[i % len(PALETTE)]
        on = np.asarray(mask) > 0
        if on.shape != (h, w):   # defensive: model output should already be full-size
            on = np.asarray(Image.fromarray(on.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        alpha = Image.fromarray((on * int(255 * MASK_ALPHA)).astype(np.uint8))
        layer = Image.new("RGBA", (w, h), color + (0,))
        layer.putalpha(alpha)
        canvas = Image.alpha_composite(canvas, layer)

    canvas = canvas.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        color = PALETTE[i % len(PALETTE)]
        x0, y0, x1, y1 = (float(v) for v in box)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line)

        tag = f"#{i} {float(score):.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        tw, th = right - left, bottom - top
        tag_y = y0 - th - 8 if y0 - th - 8 >= 0 else y0 + 2
        draw.rectangle([x0, tag_y, x0 + tw + 8, tag_y + th + 6], fill=(0, 0, 0))
        draw.text((x0 + 4, tag_y + 2 - top), tag, fill=color, font=font)

    banner_h = font_px * 2 + 8
    out = Image.new("RGB", (w, h + banner_h), (25, 25, 25))
    out.paste(canvas, (0, banner_h))
    ImageDraw.Draw(out).text(
        (10, banner_h // 2 - font_px // 2 - 2),
        f'prompt: "{prompt}"   |   {len(masks)} found',
        fill=(255, 255, 255),
        font=font,
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Visualize SAM3 output for several text prompts (no filtering).")
    ap.add_argument("--images_dir", default=str(HERE / "images"), help="Folder of test frames")
    ap.add_argument("--out_dir", default=str(HERE / "visualization"), help="Where overlays are written")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="Text prompts to try")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="SAM3 instance confidence threshold (default 0.3; the real pipeline uses 0.5)")
    ap.add_argument("--mask_threshold", type=float, default=0.5, help="Mask binarisation threshold")
    ap.add_argument("--device", default="", help="Override device (cuda:0, mps, cpu). Default: cuda > mps > cpu")
    args = ap.parse_args(argv)

    images = find_images(args.images_dir)
    if not images:
        print(f"No .png/.jpg/.jpeg images found in {args.images_dir}")
        return 1

    out_dir = Path(args.out_dir)
    prompts = list(args.prompts)
    print(f"{len(images)} image(s) x {len(prompts)} prompt(s) -> {out_dir}")
    print(f"threshold={args.threshold}  mask_threshold={args.mask_threshold}\n")

    device = pick_device(args.device)
    model, processor = load_model(device)

    stem_counts = Counter(p.stem for p in images)
    n_ok = n_fail = 0
    for img_path in images:
        # One folder per image. If two files share a stem (a.png and a.jpg), both get the suffix.
        stem = img_path.stem
        folder_name = stem if stem_counts[stem] == 1 else f"{stem}_{img_path.suffix.lstrip('.').lower()}"
        folder = out_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        image = Image.open(img_path).convert("RGB")
        print(f"=== {img_path.name} ({image.width}x{image.height}) ===")

        for idx, prompt in enumerate(prompts, 1):
            out_path = folder / f"{idx:02d}_{slugify(prompt)}.png"
            try:
                masks, boxes, scores = segment(image, prompt, model, processor, device,
                                               args.threshold, args.mask_threshold)
                render_overlay(image, masks, boxes, scores, prompt).save(out_path)
            except Exception:
                traceback.print_exc()
                print(f"  [{idx}/{len(prompts)}] {prompt!r}: FAILED (continuing)")
                n_fail += 1
                continue

            areas = [int((np.asarray(m) > 0).sum()) for m in masks]
            print(f"  [{idx}/{len(prompts)}] {prompt!r}: {len(masks)} found"
                  f"  scores={[round(float(s), 2) for s in scores]}  area_px={areas}")
            n_ok += 1
        print()

    print(f"Done: {n_ok} overlay(s) written, {n_fail} failed. Open {out_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())