import time
import torch
import cv2
import numpy as np
from utils import (
    keep_largest_component,
    get_labeled_corners,
    get_labeled_corners_area,
    get_part_boxes,
)
from post_selection import (
    select_crossbar,
    select_posts_area,
    select_posts_cluster,
    select_posts_pairwise,
)

select_posts = select_posts_pairwise

confidence_threshold = 0.3

def process_frame(image, processor, model, prompts, device):
    inference_start = time.time()

    # Setup empty lists to accumulate results across all prompts
    all_masks = []
    all_boxes = []
    all_scores = []

    # Loop through each prompt
    for prompt in prompts:
        print(f"  -> Processing: '{prompt}'")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=confidence_threshold,
            mask_threshold=confidence_threshold,
            target_sizes=[image.size[::-1]], 
        )[0]
        
        # Only append if the model actually found a matching mask for this specific prompt
        if len(results["masks"]) > 0:
            masks = results["masks"]
            boxes = results["boxes"]
            scores = results["scores"]
            
            # Convert tensor masks to numpy arrays for OpenCV
            masks_np = masks.cpu().numpy().astype(np.uint8)
            scores_np = scores.detach().float().cpu().numpy()
            valid_indices = select_posts(masks_np, scores_np)

            if len(valid_indices) == 0:
                print("    -> No instance passed post selection for this prompt")
                continue

            print(f"    -> Selected candidates {[int(i) for i in valid_indices]} of {len(masks)}, "
                  f"scores {[round(float(scores_np[i]), 2) for i in valid_indices]}")

            # Filter the tensors
            filtered_masks = masks[valid_indices]
            filtered_boxes = boxes[valid_indices]
            filtered_scores = scores[valid_indices]

            cleaned_masks = []
            cleaned_boxes = []
            
            for k, (mask, box) in enumerate(zip(filtered_masks, filtered_boxes)):
                c_mask, c_box = keep_largest_component(mask, box)
                cleaned_masks.append(c_mask)
                cleaned_boxes.append(c_box)

                # Log what cleaning did: a tall post shrinking to a small blob shows up here
                before = int((mask.cpu().numpy() > 0).sum())
                after = int((c_mask.cpu().numpy() > 0).sum())
                print(f"    -> mask {k}: {before} -> {after} px after cleaning, box {[int(v) for v in c_box.tolist()]}")
                if before > 0 and after < 0.5 * before:
                    print(f"    !! cleaning removed {100 * (1 - after / before):.0f}% of mask {k}")
                
            filtered_masks = torch.stack(cleaned_masks)
            filtered_boxes = torch.stack(cleaned_boxes)
            
            all_masks.append(filtered_masks)
            all_boxes.append(filtered_boxes)
            all_scores.append(filtered_scores)
    
    inference_end = time.time()
    elapsed_time = inference_end - inference_start

    print(f"Total sequential execution time: {elapsed_time:.4f} seconds")

    # Combine all collected data and draw
    if all_masks:
        combined_masks = torch.cat(all_masks, dim=0)
        combined_boxes = torch.cat(all_boxes, dim=0)
        combined_scores = torch.cat(all_scores, dim=0)
        
        print(f"Found {len(torch.cat(all_masks, dim=0))} instance(s) combined")
        print(f"Scores: {combined_scores.tolist()}")
        print("Keypoints stored sucessfully")

        corners = get_labeled_corners_area(combined_masks, image.width)
        print("Corners: " + str({k: None if v is None else (round(float(v[0])), round(float(v[1]))) for k, v in corners.items()}))
        return corners, combined_masks
    else:
        print("No instances found for any of the provided prompts.")
        return None, None

# ---------------------------------------
#   --enrich: part boxes (posts + crossbar)
# ---------------------------------------

BAR_PROMPT = "horizontal pole"

def find_crossbar_mask(image, processor, model, device, post_masks):
    """
    Second SAM3 pass with the crossbar prompt; returns the mask (numpy) that spans the two posts, or None.
    Needs two posts: the bar is defined by the posts it hangs between.
    """
    if post_masks is None or len(post_masks) < 2:
        print("  -> Crossbar: needs two posts, skipping")
        return None

    print(f"  -> Processing: '{BAR_PROMPT}'")
    inputs = processor(images=image, text=BAR_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=confidence_threshold,
        mask_threshold=confidence_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    if len(results["masks"]) == 0:
        print("    -> Crossbar: SAM3 found no horizontal pole")
        return None

    bar_masks = results["masks"].cpu().numpy().astype(np.uint8)
    bar_scores = results["scores"].detach().float().cpu().numpy()
    idx, note = select_crossbar(bar_masks, bar_scores, post_masks)
    print(f"    -> Crossbar: {note}")
    return None if idx is None else bar_masks[idx]


def process_parts(image, processor, model, device, post_masks):
    """
    Boxes for the --enrich label: left post, right post (from the cleaned post masks process_frame
    returned) and the crossbar. Returns {"left_post", "middle_divider", "right_post", "crossbar"},
    each [x_min, y_min, x_max, y_max] in pixels or None. None overall when there are no posts.
    """
    if post_masks is None:
        return None
    bar_mask = find_crossbar_mask(image, processor, model, device, post_masks)
    return get_part_boxes(post_masks, bar_mask, image.width)