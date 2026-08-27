import time
import torch
import cv2
import numpy as np
from utils import (
    keep_largest_component,
    get_labeled_corners,
)

confidence_threshold = 0.5
post_ratio_thresh = 0.5

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
            
            areas = []
            # Convert tensor masks to numpy arrays for OpenCV
            masks_np = masks.cpu().numpy().astype(np.uint8)
            
            for m in masks_np:
                # Scale 0/1 binary mask to 0/255 for cv2
                m_255 = m * 255
                contours, _ = cv2.findContours(m_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour area within this specific mask
                    max_area = max([cv2.contourArea(c) for c in contours])
                    areas.append(max_area)
                else:
                    areas.append(0.0)
                    
            # Sort indices by area in descending order
            sorted_indices = np.argsort(areas)[::-1].tolist()

            valid_indices = []
            if len(sorted_indices) > 0:

                # Always keep the largest area (the primary post)
                largest_idx = sorted_indices[0]
                largest_area = areas[largest_idx]
                
                if largest_area > 0:
                    valid_indices.append(largest_idx)
                
                # Evaluate the second largest if it exists
                if len(sorted_indices) > 1:
                    second_idx = sorted_indices[1]
                    second_area = areas[second_idx]
                    
                    # Check if the second post meets the 0.7 ratio threshold
                    if (second_area / largest_area) >= post_ratio_thresh:
                        valid_indices.append(second_idx)
                    else:
                        print(f"    -> Discarding second post. Ratio: {(second_area/largest_area):.2f} < {post_ratio_thresh}")
            
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
        combined_boxes = torch.cat(all_boxes, dim=0)
        combined_scores = torch.cat(all_scores, dim=0)
        
        print(f"Found {len(torch.cat(all_masks, dim=0))} instance(s) combined")
        print(f"Scores: {combined_scores.tolist()}")
        print("Keypoints stored sucessfully")

        return get_labeled_corners(combined_boxes, image.width)
    else:
        print("No instances found for any of the provided prompts.")
        return None