import torch
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil
import matplotlib
import csv
from pathlib import Path

# ---------------------------------------
#               BACKEND
# ---------------------------------------

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=3)
    return image

def draw_global_corners(image, combined_boxes):
    corners = get_labeled_corners(combined_boxes, image.width)
    if not corners:
        return image

    draw = ImageDraw.Draw(image)
    r = 6  # Radius of the keypoint dot

    try:
        large_font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        large_font = ImageFont.load_default(size=30)

    for label_name, (cx, cy) in corners.items():
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="cyan", outline="white", width=2)
        draw.text((cx + r + 5, cy - 8), label_name, fill="yellow", font=large_font)

    return image

def keep_largest_component(mask_tensor, original_box):
    """
    Helper function for segmentations that identify discontinuous bodies as one object.
    """
    # Convert tensor into a cv2 compatible mask
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    # Mild morpohological opening to make the reflection/post distinction clearer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)    

    # ---------------------------------------
    #       Try to isolate the waterline
    # ---------------------------------------
    # y_coords, _ = np.where(mask_np > 0)
    # if len(y_coords) > 0:
    #     y_min, y_max = np.min(y_coords), np.max(y_coords)
        
    #     row_centers = []
    #     valid_ys = []
        
    #     # Calculate the horizontal center point of the mask for each row
    #     for y in range(y_min, y_max + 1):
    #         row_xs = np.where(mask_np[y, :] > 0)[0]
    #         if len(row_xs) > 0:
    #             # Use the midpoint between the leftmost and rightmost pixel of the post
    #             center_x = (np.min(row_xs) + np.max(row_xs)) / 2.0
    #             row_centers.append(center_x)
    #             valid_ys.append(y)

    #     # Look for sudden horizontal shifts between adjacent rows
    #     if len(row_centers) > 1:
    #         # np.diff gets the pixel distance the center shifted from one row to the next
    #         shifts = np.abs(np.diff(row_centers))
            
    #         # If the center suddenly shifts by more than 5 pixels, we found the waterline
    #         kink_indices = np.where(shifts > 1)[0]
            
    #         if len(kink_indices) > 0:
    #             # Take the highest Y coordinate where a kink occurs
    #             kink_y_idx = kink_indices[0] + 1
    #             waterline_y = valid_ys[kink_y_idx]
                
    #             # Zero out everything above the waterline to permanently sever the reflection
    #             mask_np[:waterline_y, :] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np)

    # Make sure that tensors are always returned as opposed to numpy arrays
    if num_labels <= 1:
        original_np = (mask_tensor.cpu().numpy() > 0).astype(np.uint8)
        original_mask_tensor = torch.from_numpy(original_np).to(mask_tensor.device)
 
        ys, xs = np.nonzero(original_np)
        if len(xs) == 0:
            return original_mask_tensor, original_box
 
        fallback_box = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], device=mask_tensor.device)
        return original_mask_tensor, fallback_box

    # ---------------------------------------------
    #            DEPTH BASED APPROACH
    # ---------------------------------------------

    # Note: OpenCV has y = 0 at the top, increase as you go down
    bottom_edges = stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT]
    lowest_label = 1 + np.argmax(bottom_edges)

    clean_mask_np = (labels == lowest_label).astype(np.uint8)
    clean_mask_tensor = torch.from_numpy(clean_mask_np).to(mask_tensor.device)

    x_min = stats[lowest_label, cv2.CC_STAT_LEFT]
    y_min = stats[lowest_label, cv2.CC_STAT_TOP]
    width = stats[lowest_label, cv2.CC_STAT_WIDTH]
    height = stats[lowest_label, cv2.CC_STAT_HEIGHT]

    # ---------------------------------------------
    #            AREA BASED APPROACH
    # ---------------------------------------------
    # # Isolate pixle counts for each detected label
    # areas = stats[1:, cv2.CC_STAT_AREA]
    # largest_label = 1 + np.argmax(areas)

    # # Only keep the largest label
    # clean_mask_np = (labels == largest_label).astype(np.uint8)
    # clean_mask_tensor = torch.from_numpy(clean_mask_np).to(mask_tensor.device)

    # x_min = stats[largest_label, cv2.CC_STAT_LEFT]
    # y_min = stats[largest_label, cv2.CC_STAT_TOP]
    # width = stats[largest_label, cv2.CC_STAT_WIDTH]
    # height = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Draw bounding box
    clean_box = torch.tensor([x_min, y_min, x_min + width, y_min + height], device=mask_tensor.device)

    return clean_mask_tensor, clean_box

def get_labeled_corners(combined_boxes, image_width):
    """
    Calculates and returns named keypoint coordinates for dataset labeling
    """
    if len(combined_boxes) == 0:
        return {}

    if len(combined_boxes) == 1:
        x_min, y_min, x_max, y_max = combined_boxes[0].tolist()
        x_center = (x_min + x_max) / 2
        
        # Compare post center to image center
        if x_center < (image_width / 2):
            return {
                "TL": (x_center, y_min),
                "TR": None,
                "BL": (x_center, y_max),
                "BR": None,
            }
        else:
            return {
                "TL": None,
                "TR": (x_center, y_min),
                "BL": None,
                "BR": (x_center, y_max),
            }

    # Sort boxes left-to-right by x_min
    sorted_ind = torch.argsort(combined_boxes[:, 0])
    sorted_boxes = combined_boxes[sorted_ind]

    lx_min, ly_min, lx_max, ly_max = sorted_boxes[0].tolist()
    rx_min, ry_min, rx_max, ry_max = sorted_boxes[1].tolist()

    left_x_center = (lx_min + lx_max) / 2
    right_x_center = (rx_min + rx_max) / 2

    return {
        "TL": (left_x_center, ly_min),
        "TR": (right_x_center, ry_min),
        "BL": (left_x_center, ly_max),
        "BR": (right_x_center, ry_max),
    }

def get_labeled_corners_area(combined_masks, image_width):
    """
    Get labeled corners using area averaging
    """

    if len(combined_masks) == 0:
        return {}

    mask_data = []

    for mask in combined_masks:

        # CUDA Tensor -> CPU NumPy for np.where
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()

        # obtain all (y,x) coords of mask
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            continue

        y_min, y_max = np.min(y_coords), np.max(y_coords)
        post_height = y_max - y_min

        # Isolate top/bottom 5% of pixels
        region = post_height * 0.05 # consider making this a constant

        # Average top pixels
        top_pixels = y_coords <= (y_min + region)
        top_pt = (np.mean(x_coords[top_pixels]), np.mean(y_coords[top_pixels]))

        # Average bottom pixels
        bot_pixels = y_coords >= (y_max - region)
        bot_pt = (np.mean(x_coords[bot_pixels]), np.mean(y_coords[bot_pixels]))

        # Average center of post for sorting purposes
        center_x = np.mean(x_coords)

        mask_data.append({
            "center_x": center_x,
            "top": top_pt,
            "bottom": bot_pt,
        })

    if len(mask_data) == 0:
        return {}

    # Sort by x
    mask_data = sorted(mask_data, key=lambda d: d["center_x"])    

    # Single post case    
    if len(mask_data) == 1:
        post = mask_data[0]
        if post["center_x"] < (image_width / 2):
            return {
                "TL": post["top"],
                "TR": None,
                "BL": post["bottom"],
                "BR": None,
            }
        else:
            return {
                "TL": None,
                "TR": post["top"],
                "BL": None,
                "BR": post["bottom"],
            }

    # Gate in full view
    left_post = mask_data[0]
    right_post = mask_data[-1]

    return {
        "TL": left_post["top"],
        "TR": right_post["top"],
        "BL": left_post["bottom"],
        "BR": right_post["bottom"],
    }


# ---------------------------------------
#               FRONTEND
# ---------------------------------------

def load_predictions(json_path):
    """
    Load raw_predictions.json for review:

    Treat null readings as empty predictions in case of faulty
    null detections => let the reviewer draw the kpts
    """
    with open(json_path, "r") as f:
        raw = json.load(f)
    return {filename: (kpts if isinstance(kpts, dict) else {}) for filename, kpts in raw.items()}


def draw_JSON_kpts(img_path, kpts: dict):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    r = 16
    for label, coords in kpts.items():
        if coords is not None:
            x, y = coords
            draw.ellipse([x - r, y - r, x + r, y + r], fill="cyan", outline="black", width=2)
            draw.text((x + r + 5, y - 10), label, fill="yellow")
    return img

def missing_sides(points):
    """Sides of the gate where BOTH corners are missing"""
    def missing(a, b):
        return points.get(a) is None and points.get(b) is None
 
    sides = []
    if missing("TL", "BL"):
        sides.append("left")
    if missing("TR", "BR"):
        sides.append("right")
    if missing("TL", "TR"):
        sides.append("top")
    if missing("BL", "BR"):
        sides.append("bottom")
    return sides


def compute_gate_box(points, img_width, img_height, allow_truncated=False):
    """
    Return bounding box (x_min, y_min, x_max, y_max) in pixels for the YOLO label, or None.

    A gate with only one post (both left corners or both right corners missing):
        allow_truncated=True  -> the box is extended to the image border on the side
                                where the gate runs off-frame (YOLO-pose can use this,
                                the missing corners just get visibility 0)
        allow_truncated=False -> returns None (a whole-gate box can't be formed)

    Returns None when there are no visible corners or the box would have no area.
    post is visible.
    """

    visible = {name: pt for name, pt in points.items() if pt is not None}

    if not visible:
        return None

    xs = [pt[0] for pt in visible.values()]
    ys = [pt[1] for pt in visible.values()]
 
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    sides = missing_sides(points)

    if "left" in sides or "right" in sides:
        if not allow_truncated:
            return None
        if "left" in sides:
            x_min = 0
        if "right" in sides:
            x_max = img_width

    if "top" in sides:
        y_min = 0
    if "bottom" in sides:
        y_max = img_height

    # Keep the box inside the image
    x_min, x_max = max(0, x_min), min(img_width, x_max)
    y_min, y_max = max(0, y_min), min(img_height, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None
 
    return x_min, y_min, x_max, y_max

def export_to_yolo(filename, points, img_width, img_height, labels_dir, class_id=0, allow_truncated=True):

    # Create the labels directory => Consider moving this elsewhere
    labels_dir.mkdir(parents=True, exist_ok=True)

    txt_fn = Path(filename).with_suffix('.txt').name
    txt_path = labels_dir / txt_fn

    box = compute_gate_box(points, img_width, img_height, allow_truncated=allow_truncated)

    if box is None:
        if any(coords is not None for coords in points.values()):
            print(f"Skipped {filename}: visible corners do not form a valid box")
        if txt_path.exists():
            txt_path.unlink()
        return

    # Obtain the yolo-pose string
    x_min, y_min, x_max, y_max = box
 
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height


    kp_str = ""
    for corner in ["TL", "TR", "BL", "BR"]:
        coords = points.get(corner)

        if coords is not None:
            nx = coords[0] / img_width
            ny = coords[1] / img_height
            kp_str += f"{nx:.6f} {ny:.6f} 2 "

        # For non-visible keypoints
        else:
            kp_str += "0.00000 0.00000 0 "

    final_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {kp_str.strip()}"

    # Write a new file to this directory for each set of labels

    with open(txt_path, "w") as f:
        f.write(final_line)

    print(f"Saved: {txt_path}")

def export_no_gate_pose(filename, labels_dir):
    """Handle empty YOLO-pose label AFTER reviewer confirmation"""
    labels_dir.mkdir(parents=True, exist_ok=True)
    txt_path = labels_dir / Path(filename).with_suffix('.txt').name
    txt_path.write_text("")
    print(f"Saved (no gate): {txt_path}")    

# ---------------------------------------
#   INTRO PROJECT EXPORT (YOLO-DETECT)
# ---------------------------------------
# The intro perception project (berkeleyauv/intro-perception) trains and scores a
# plain YOLO *detect* model whose class is the gate's orientation. These helpers
# derive that class from the same four corners used for YOLO-pose, so the corners
# stay the single source of truth and the pose export is untouched.
#
# Output layout (what intro-perception's `split` expects): one flat folder with
# <name>.<ext> and <name>.txt side by side, plus manifest.csv with every decision.
 
CORNERS = ("TL", "TR", "BL", "BR")
 
GATE_CLASS_ID = 0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
 
MANIFEST_FIELDS = [
    "filename", "status", "reason", "truncated", "n_corners",
    "box_cx", "box_cy", "box_w", "box_h",
]
 
def describe_gate(points, skip_truncated=False):
    """One-line summary of how the current corners will export, for the review UI."""
    n_corners = sum(1 for corner in CORNERS if points.get(corner) is not None)
 
    if n_corners == 0:
        return "Gate: **no corners yet** | click the 4 corners, or use No Gate if the gate is not in this image"
 
    sides = missing_sides(points)
 
    if not sides:
        note = "" if n_corners == 4 else " (1 corner missing, the box uses the visible ones)"
        return f"Gate: **whole gate** | {n_corners}/4 corners{note}"
 
    if "left" in sides or "right" in sides:
        gone = "left" if "left" in sides else "right"
        present = "right" if gone == "left" else "left"
        if skip_truncated:
            return f"Gate: **single post** (only the {present} post) | will be SKIPPED (--skip_truncated)"
        return f"Gate: **single post** (only the {present} post) | box will extend to the {gone} image edge"
 
    edges = " and ".join(sides)
    return f"Gate: both posts visible, {edges} corners missing | box will extend to the {edges} image edge" 
 
def update_manifest(manifest_path, row):
    """Insert or replace this image's row, so re-reviewing an image never duplicates it."""
    manifest_path = Path(manifest_path)
    rows = {}
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            rows = {r["filename"]: r for r in csv.DictReader(f)}
 
    rows[row["filename"]] = row
 
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for filename in sorted(rows):
            writer.writerow(rows[filename])

def reviewed_stems(labels_dir, detect_dir):
    """
    Image names (without extension) that an earlier QA session already dealt with.
 
    An image counts as reviewed if it has a pose label in labels_dir, or any row in the
    detect manifest (including skipped ones), so --resume never shows it again and can't
    overwrite a corrected label with the original prediction.
    """
    done = set()
 
    labels_dir, detect_dir = Path(labels_dir), Path(detect_dir)
    if labels_dir.is_dir():
        done.update(path.stem for path in labels_dir.glob("*.txt"))
 
    manifest_path = detect_dir / "manifest.csv"
    if manifest_path.is_file():
        with open(manifest_path, newline="") as f:
            done.update(Path(row["filename"]).stem for row in csv.DictReader(f))
 
    return done
 
def _remove_exported(out_dir, stem):
    """Delete an earlier export of this image so a re-review can't leave a stale label."""
    for path in out_dir.iterdir():
        if path.stem == stem and (path.suffix.lower() in IMAGE_SUFFIXES or path.suffix == ".txt"):
            path.unlink()
 
def _place_image(src, dst, mode):
    if mode == "link":
        try:
            dst.symlink_to(Path(src).resolve())
            return
        except OSError:
            pass  # e.g. symlinks not permitted: fall back to a copy
    shutil.copy2(src, dst)
 
def export_to_intro_detect(filename, points, img_path, img_width, img_height, out_dir,
                           no_gate=False, allow_truncated=True, images="none"):
    """
    Write one YOLO-detect label (single class, 0 = gate) and log the decision in manifest.csv.
 
    The box is the bounding box of the visible corners. A gate with one post missing is
    kept by default, with the box extended to the image edge on the side where it runs
    off-frame (allow_truncated=False skips those instead). Anything with no usable box
    is skipped: nothing is written and the reason goes in the manifest.
 
    no_gate=True writes an EMPTY label, meaning "no gate in this image" (a confirmed
    negative). Only use it when the reviewer explicitly says so.
 
    images: "none" (labels only), or "copy" / "link" to also place the image next to its label.
 
    Returns the manifest row (a dict) so the caller can show what happened.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
 
    n_corners = sum(1 for corner in CORNERS if points.get(corner) is not None)
    sides = missing_sides(points)
 
    row = {"filename": filename, "status": "", "reason": "", "truncated": "", "n_corners": n_corners,
           "box_cx": "", "box_cy": "", "box_w": "", "box_h": ""}
 
    # A re-review replaces the earlier decision completely
    _remove_exported(out_dir, stem)
 
    if no_gate:
        label_text = ""
        row.update(status="no_gate", reason="reviewer marked no gate")
    else:
        box = compute_gate_box(points, img_width, img_height, allow_truncated=allow_truncated)
 
        if box is None:
            if n_corners == 0:
                reason = "no corners (use No Gate if the image has no gate)"
            elif ("left" in sides or "right" in sides) and not allow_truncated:
                reason = "single post (truncated gates are skipped)"
            else:
                reason = "corners enclose no area"
            row.update(status="skipped", reason=reason, truncated=bool(sides) if n_corners else "")
            update_manifest(out_dir / "manifest.csv", row)
            print(f"Skipped (detect) {filename}: {reason}")
            return row
 
        x_min, y_min, x_max, y_max = box
        cx = ((x_min + x_max) / 2) / img_width
        cy = ((y_min + y_max) / 2) / img_height
        bw = (x_max - x_min) / img_width
        bh = (y_max - y_min) / img_height
 
        label_text = f"{GATE_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        row.update(status="exported", truncated=bool(sides),
                   box_cx=f"{cx:.6f}", box_cy=f"{cy:.6f}", box_w=f"{bw:.6f}", box_h=f"{bh:.6f}")
 
    (out_dir / f"{stem}.txt").write_text(label_text)
 
    if images != "none":
        _place_image(img_path, out_dir / Path(filename).name, images)
 
    update_manifest(out_dir / "manifest.csv", row)
    print(f"Saved (detect): {out_dir / (stem + '.txt')}  [{row['status']}]")
    return row
