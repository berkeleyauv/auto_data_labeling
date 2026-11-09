import time
import torch
import cv2
import numpy as np
from utils import (
    keep_largest_component,
    get_labeled_corners,
    get_labeled_corners_area,
)
from post_selection import (
    select_posts_area,
    select_posts_cluster,
    select_posts_pairwise,
)

select_posts = select_posts_cluster

confidence_threshold = 0.5

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
            valid_indices = select_posts(masks_np, scores.detach().float().cpu().numpy())

            if len(valid_indices) == 0:
                print("    -> No instance passed post selection for this prompt")
                continue

            # Filter the tensors
            filtered_masks = masks[valid_indices]
            filtered_boxes = boxes[valid_indices]
            filtered_scores = scores[valid_indices]

            cleaned_masks = []
            cleaned_boxes = []
            
            for mask, box in zip(filtered_masks, filtered_boxes):
                c_mask, c_box = keep_largest_component(mask, box)
                cleaned_masks.append(c_mask)
                cleaned_boxes.append(c_box)
                
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

        return get_labeled_corners_area(combined_masks, image.width), combined_masks
    else:
        print("No instances found for any of the provided prompts.")
        return None, None