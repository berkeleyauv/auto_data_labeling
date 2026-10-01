import torch
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil
import math
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

    # Ignore if there is no separation
    if num_labels <= 1:
        return mask_np, original_box

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

def compute_gate_box(points, img_width, img_height, allow_truncated=False):
    """
    Return bounding box (x_min, y_min, x_max, y_max) in pixels for the YOLO label, or None.

    A gate with only one post (both left corners or both right corners missing):
        allow_truncated=True  -> the box is extended to the image border on the side
                                where the gate runs off-frame (YOLO-pose can use this,
                                the missing corners just get visibility 0)
        allow_truncated=False -> returns None (a whole-gate box can't be formed)

    JUST FOR INTRO PROJECT: Return None when there are no visible corners or if only one
    post is visible.
    """

    visible = {name: pt for name, pt in points.items() if pt is not None}

    if not visible:
        return None

    xs = [pt[0] for pt in visible.values()]
    ys = [pt[1] for pt in visible.values()]
 
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    def both_missing(a, b):
        return a not in visible and b not in visible

    # Return None for single post gates
    if both_missing("TL", "BL") or both_missing("TR", "BR"):
        if not allow_truncated:
            return None
        if both_missing("TL", "BL"):
            x_min = 0
        if both_missing("TR", "BR"):
            x_max = img_width

    # Edge cases
    if both_missing("TL", "TR"):
        y_min = 0
    if both_missing("BL", "BR"):
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
 
# Class ids used by the intro project:
#   0 = left     the gate's right side appears closer (right post taller)
#   1 = head_on  both posts look about the same height
#   2 = right    the gate's left side appears closer (left post taller)
INTRO_CLASSES = {"left": 0, "head_on": 1, "right": 2}
 
# PLACEHOLDERS: calibrate on a hand-labeled sample before trusting them.
# ratio = right edge length / left edge length.
#   spread <= HEAD_ON  -> head_on
#   spread >= CLEAR    -> left / right
#   in between         -> ambiguous (skipped, as the intro GUIDE says to do)
# where spread = max(ratio, 1/ratio), so left and right are treated symmetrically.
DEFAULT_HEAD_ON_RATIO = 1.06
DEFAULT_CLEAR_RATIO = 1.25
 
MANIFEST_FIELDS = [
    "filename", "status", "reason", "class_name", "class_id",
    "ratio", "left_edge_px", "right_edge_px", "head_on_ratio", "clear_ratio",
]
 
INTRO_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
 
 
def compute_orientation(points, head_on_ratio=DEFAULT_HEAD_ON_RATIO, clear_ratio=DEFAULT_CLEAR_RATIO):
    """
    Orientation class from the four corners, using the lengths of the two vertical edges.
 
    Needs all four corners (each edge needs both of its ends). Edge lengths are
    Euclidean, so a small camera roll doesn't change them.
 
    Returns a dict: class_name / class_id (None when no class can be assigned),
    ratio (right edge / left edge), left_edge, right_edge (px), reason (why not).
    """
    if not 1.0 <= head_on_ratio < clear_ratio:
        raise ValueError("thresholds must satisfy 1 <= head_on_ratio < clear_ratio")
 
    result = {"class_name": None, "class_id": None, "ratio": None,
              "left_edge": None, "right_edge": None, "reason": ""}
 
    missing = [c for c in CORNERS if points.get(c) is None]
    if missing:
        result["reason"] = "missing corner(s): " + ", ".join(missing)
        return result
 
    left_edge = math.dist(points["TL"], points["BL"])
    right_edge = math.dist(points["TR"], points["BR"])
    result["left_edge"], result["right_edge"] = left_edge, right_edge
 
    if left_edge < 1 or right_edge < 1:
        result["reason"] = "degenerate post (edge shorter than 1 px)"
        return result
 
    ratio = right_edge / left_edge
    result["ratio"] = ratio
    spread = max(ratio, 1 / ratio)
 
    if spread <= head_on_ratio:
        name = "head_on"
    elif spread >= clear_ratio:
        name = "left" if ratio > 1 else "right"
    else:
        result["reason"] = (f"ambiguous: ratio {ratio:.3f} falls between the head-on "
                            f"and clear bands ({head_on_ratio}/{clear_ratio})")
        return result
 
    result["class_name"] = name
    result["class_id"] = INTRO_CLASSES[name]
    return result
 
 
def describe_orientation(points, head_on_ratio=DEFAULT_HEAD_ON_RATIO, clear_ratio=DEFAULT_CLEAR_RATIO):
    """One-line summary of compute_orientation for the review UI."""
    o = compute_orientation(points, head_on_ratio, clear_ratio)
 
    if o["class_name"] is not None:
        return (f"Orientation: **{o['class_name']}** (class {o['class_id']}) | "
                f"ratio {o['ratio']:.3f} (right edge {o['right_edge']:.0f}px / left edge {o['left_edge']:.0f}px)")
    if o["ratio"] is not None:
        return (f"Orientation: **ambiguous** | ratio {o['ratio']:.3f} is between {head_on_ratio} and "
                f"{clear_ratio}, so this image is skipped in the intro export")
    return f"Orientation: n/a | {o['reason']}"
 
 
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
 
 
def _remove_exported(out_dir, stem):
    """Delete an earlier export of this image so a re-review can't leave a stale label."""
    for path in out_dir.iterdir():
        if path.stem == stem and (path.suffix.lower() in INTRO_IMAGE_SUFFIXES or path.suffix == ".txt"):
            path.unlink()
 
 
def _place_image(src, dst, link=False):
    if link:
        try:
            dst.symlink_to(Path(src).resolve())
            return
        except OSError:
            pass  # e.g. symlinks not permitted: fall back to a copy
    shutil.copy2(src, dst)
 
 
def export_to_intro_detect(filename, points, img_path, img_width, img_height, out_dir,
                           head_on_ratio=DEFAULT_HEAD_ON_RATIO, clear_ratio=DEFAULT_CLEAR_RATIO,
                           no_gate=False, link_images=False):
    """
    Write one image + its YOLO-detect label for the intro project, and log the decision.
 
    Exported only when all four corners are visible and the orientation is clear.
    Everything else is skipped (nothing written) and the reason goes in manifest.csv.
    no_gate=True writes the image with an EMPTY label (a confirmed negative); this is
    only ever used when the reviewer explicitly says there is no gate.
 
    Returns the manifest row (a dict) so the caller can show what happened.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
 
    row = {"filename": filename, "status": "", "reason": "", "class_name": "", "class_id": "",
           "ratio": "", "left_edge_px": "", "right_edge_px": "",
           "head_on_ratio": head_on_ratio, "clear_ratio": clear_ratio}
 
    # A re-review replaces the earlier decision completely
    _remove_exported(out_dir, stem)
 
    if no_gate:
        label_text = ""
        row.update(status="no_gate", reason="reviewer marked no gate")
    else:
        o = compute_orientation(points, head_on_ratio, clear_ratio)
        if o["ratio"] is not None:
            row.update(ratio=f"{o['ratio']:.4f}", left_edge_px=f"{o['left_edge']:.1f}",
                       right_edge_px=f"{o['right_edge']:.1f}")
 
        box = None
        if o["class_id"] is not None:
            box = compute_gate_box(points, img_width, img_height, allow_truncated=False)
 
        if o["class_id"] is None or box is None:
            row.update(status="skipped", reason=o["reason"] or "corners enclose no area")
            update_manifest(out_dir / "manifest.csv", row)
            print(f"Skipped (intro detect) {filename}: {row['reason']}")
            return row
 
        x_min, y_min, x_max, y_max = box
        label_text = (f"{o['class_id']} {((x_min + x_max) / 2) / img_width:.6f} "
                      f"{((y_min + y_max) / 2) / img_height:.6f} "
                      f"{(x_max - x_min) / img_width:.6f} {(y_max - y_min) / img_height:.6f}\n")
        row.update(status="exported", class_name=o["class_name"], class_id=o["class_id"])
 
    (out_dir / f"{stem}.txt").write_text(label_text)
    _place_image(img_path, out_dir / Path(filename).name, link=link_images)
    update_manifest(out_dir / "manifest.csv", row)
    print(f"Saved (intro detect): {out_dir / (stem + '.txt')}  [{row['status']}]")
    return row
