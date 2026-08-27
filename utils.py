import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor
import numpy as np
import matplotlib
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
    r = 12  # Radius of the keypoint dot

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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np)

    # Ignore if there is no separation
    if num_labels <= 1:
        return mask_np, original_box

    # Isolate pixle counts for each detected label
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    # Only keep the largest label
    clean_mask_np = (labels == largest_label).astype(np.uint8)
    clean_mask_tensor = torch.from_numpy(clean_mask_np).to(mask_tensor.device)

    x_min = stats[largest_label, cv2.CC_STAT_LEFT]
    y_min = stats[largest_label, cv2.CC_STAT_TOP]
    width = stats[largest_label, cv2.CC_STAT_WIDTH]
    height = stats[largest_label, cv2.CC_STAT_HEIGHT]

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

# ---------------------------------------
#               FRONTEND
# ---------------------------------------

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

def export_to_yolo(filename, points, img_width, img_height, labels_dir, class_id=0):

    # Create the labels directory => Consider moving this elsewhere
    labels_dir.mkdir(parents=True, exist_ok=True)

    valid_points = [coords for coords in points.values() if coords is not None]

    if not valid_points:
        return

    # Obtain the yolo-pose string
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

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
    txt_fn = Path(filename).with_suffix('.txt').name
    txt_path = labels_dir / txt_fn

    with open(txt_path, "w") as f:
        f.write(final_line)

    print(f"Saved: {txt_path}")
    