import json
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor
from pathlib import Path
import time
from annotate_frame import process_frame

def process_dataset(input_dir, output_dir, device_str):

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "raw_predictions.json"

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

    prompts = ["vertical black pole"] # "horizontal black pole"

    print(f"Running sequential processing for {[p for p in prompts]}")

    preds_dct = {}

    inference_start = time.time()

    # Loop through each image
    for idx, path in enumerate(img_paths, 1):
        print(f"\n --- [{idx}/{len(img_paths)}] Processing {path.name} ---")

        image = Image.open(path).convert("RGB")

        keypoints = process_frame(image, processor, model, prompts, device)

        preds_dct[path.name] = keypoints

        print(f"--- Done with image {idx} ---")

    inference_end = time.time()

    print(f"\nSaving raw predictions to {json_path}")
    with open(json_path, "w") as f:
        json.dump(preds_dct, f, indent=4)

    print(f"Dataset processing complete, ran in {inference_end - inference_start:.2f} seconds")
    return json_path

if __name__ == "__main__":

    # Setup paths for annotations and exports
    parser = argparse.ArgumentParser(description="SAM3 Labeling Inference Runner")
    parser.add_argument("--input_dir", type=str, default="./Test_Images", help="Path to raw images")
    parser.add_argument("--output_dir", type=str, default="./Test_JSON", help="Path to saved prediction JSON")
    parser.add_argument("--device", type=str, default="", help="Override torch device (eg. cuda:0, cpu)")

    args= parser.parse_args()
    process_dataset(Path(args.input_dir), Path(args.output_dir), args.device)