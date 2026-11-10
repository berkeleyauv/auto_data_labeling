import json
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor
from pathlib import Path
import time
import os
import traceback
from annotate_frame import process_frame, process_parts
from utils import overlay_masks, draw_parts

def save_json(path, data):
    """Save json in the event of a crash"""
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)

def process_dataset(input_dir, output_dir, device_str, visualize=False, resume=False, save_every=25, enrich=False):

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "raw_predictions.json"
    parts_path = output_dir / "raw_parts.json"      # --enrich only; raw_predictions.json is unchanged

    vis_dir = output_dir / "visualizations"
    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Filter out non-valid image formats
    valid_extensions = {".png", ".jpg", ".jpeg"}
    img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]

    # Load model weights ONCE
    if device_str:
        device = device_str
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Loading model weights into memory...")
    load_start = time.time()

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    load_end = time.time()
    print(f"Model loaded in: {load_end - load_start:.2f} seconds\n")

    prompts = ["vertical pole"]

    print(f"Running sequential processing for {[p for p in prompts]}")

    preds_dct = {}
    parts_dct = {}

    # pick up where an earlier (crashed / timed-out) run stopped
    if resume and json_path.exists():
        with open(json_path, "r") as f:
            preds_dct = json.load(f)
        print(f"Resuming: {len(preds_dct)} images already done, skipping them")
        if enrich and parts_path.exists():
            with open(parts_path, "r") as f:
                parts_dct = json.load(f)
 
    failed = {}       # filename -> error, written to failed_images.json
    newly_done = 0

    inference_start = time.time()

    # Loop through each image
    for idx, path in enumerate(img_paths, 1):
        
        # with --enrich an image is only done once its parts are saved too
        if path.name in preds_dct and (not enrich or path.name in parts_dct):
            continue
        
        print(f"\n --- [{idx}/{len(img_paths)}] Processing {path.name} ---")

        # Don't let a bad image ruin the entire run
        try:
            image = Image.open(path).convert("RGB")
 
            keypoints, raw_masks = process_frame(image, processor, model, prompts, device)

            # --enrich: boxes for the posts and the crossbar. A failure here must not lose the corners.
            parts = None
            if enrich:
                try:
                    parts = process_parts(image, processor, model, device, raw_masks)
                except Exception:
                    traceback.print_exc()
                    print(f"!!! Parts failed on {path.name}; its corners are kept, its parts are saved as none")
 
            # Debug overlay. There are no masks when nothing was found, so check first.
            if visualize and raw_masks is not None:
                overlay = overlay_masks(image, raw_masks).convert("RGB")
                draw = ImageDraw.Draw(overlay)
                if parts:
                    draw_parts(draw, parts)
                for label, coords in (keypoints or {}).items():
                    if coords is not None:
                        x, y = coords
                        draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill="cyan", outline="black", width=2)
                        draw.text((x + 16, y - 10), label, fill="yellow")
                overlay.save(vis_dir / f"{path.stem}.jpg")
 
        except Exception as e:
            traceback.print_exc()
            failed[path.name] = repr(e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"!!! Failed on {path.name}, continuing (it is NOT saved; rerun with --resume to retry)")
            continue
 
        # ALWAYS record the result. None means nothing was found; it is saved as null so
        # the review tool still shows the image and lets a human add the corners.
        preds_dct[path.name] = keypoints
        if enrich:
            parts_dct[path.name] = parts
        newly_done += 1
 
        # Save progress regularly so a crash or a Slurm time limit only costs the last few images
        if newly_done % save_every == 0:
            save_json(json_path, preds_dct)
            if enrich:
                save_json(parts_path, parts_dct)
            print(f"[checkpoint] {len(preds_dct)} images saved")
 
        print(f"--- Done with image {idx} ---")

    inference_end = time.time()

    print(f"\nSaving raw predictions to {json_path}")
    save_json(json_path, preds_dct)
    if enrich:
        print(f"Saving part boxes to {parts_path}")
        save_json(parts_path, parts_dct)

    failed_path = output_dir / "failed_images.json"
    if failed:
        save_json(failed_path, failed)
        print(f"WARNING: {len(failed)} image(s) failed, listed in {failed_path}. Rerun with --resume to retry them.")
    elif failed_path.exists():
        failed_path.unlink()    # an earlier run's failures were all fixed on this one
 
    print(f"Dataset processing complete, ran in {inference_end - inference_start:.2f} seconds")
    return json_path

if __name__ == "__main__":

    # Setup paths for annotations and exports
    parser = argparse.ArgumentParser(description="SAM3 Labeling Inference Runner")
    parser.add_argument("--input_dir", type=str, default="./data/raw_images", help="Path to raw images")
    parser.add_argument("--output_dir", type=str, default="./data/predictions", help="Path to saved prediction JSON")
    parser.add_argument("--device", type=str, default="", help="Override torch device (eg. cuda:0, cpu)")
    parser.add_argument("--visualize", action="store_true", help="Also save mask/corner overlays to <output_dir>/visualizations")
    parser.add_argument("--resume", action="store_true", help="Continue from an existing raw_predictions.json, skipping images already done")
    parser.add_argument("--save_every", type=int, default=25, help="Save raw_predictions.json every N newly processed images")
    parser.add_argument("--enrich", action="store_true", help="Also collect part boxes (left/right post, crossbar) into <output_dir>/raw_parts.json for the enrich label. Adds a second SAM3 pass per image")

    args= parser.parse_args()
    process_dataset(Path(args.input_dir), Path(args.output_dir), args.device, args.visualize, args.resume, args.save_every, args.enrich)