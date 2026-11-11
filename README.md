# auto_data_labeling

Automated segmentation and bounding box labeling pipeline using SAM3. This automatically generates normalized YOLO-pose keypoint data for underwater gate detection. The same reviewed corners can also be exported as plain YOLO-detect labels (class = gate orientation) for the intro perception project.

## Prerequisites
1. **Compute Environment:** 
This pipeline supports CUDA, MPS, and CPU, but running it on a GPU cluster is highly recommended for optimal inferencing speeds (**Note**: The instructions below are tailored specifically for **cluster** setup)

## Installation
**1.**
Clone this auto-labeling repository and navigate into it:
```bash
git clone https://github.com/berkeleyauv/auto_data_labeling.git
cd auto_data_labeling
```

**2.**
Create and activate the virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3.**
Install the necessary dependencies:
```bash
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install repository dependencies
pip install -r requirements.txt
```

## Hugging Face Authentication
You will need a valid Hugging Face access token to download the model weights. Make sure your virtual environment is **activated** or the following instructions will error. Run the following command in your terminal and follow the prompts to activate your token: 
```bash 
hf auth login
```

If hf is not found in your path, fall back to:
```bash
huggingface-cli login
```

🚨 NOTE: Slurm jobs run non-interactively! Make sure that you login before submitting any jobs! 🚨

## Dataset Specs & Standards
**1. Folder Structure:** 
All datasets live under `./data/` and should maintain this 3-subfolder directory layout: 

```text
data/your_dataset_name/
├── raw_imgs/         # Original input frames (.jpg, .jpeg, .png)
├── predictions/       # Raw model outputs (raw_predictions.json, plus raw_parts.json with --enrich; plus visualizations/ and failed_images.json when produced)
└── labels/           # Final exported annotations (.txt files)
└── intro_detect/     # Single-class YOLO-detect labels + manifest.csv (only with --format detect/both)
└── labels_enrich/    # One-line gate + part-box labels (only with --enrich, see Enrich Export below)
```

**2. Class IDs**
* `0`: `gate` is the only class, in both the YOLO-pose labels (`labels/`) and the YOLO-detect labels (`intro_detect/`). There is no orientation class.

**3. Keypoint Ordering (YOLO-Pose):**
When reviewing and exporting keypoints, adhere to the 4-corner index convention:
* Index 0 (1st point): Top-Left (`TL`)
* Index 1 (2nd point): Top-Right (`TR`)
* Index 2 (3rd point): Bottom-Left (`BL`)
* Index 3 (4th point): Bottom-Right (`BR`)

**4. Label Format:**
Each exported `.txt` file in `labels/` follows the normalized YOLO-pose format

```text
<class-id> <x_center> <y_center> <width> <height> <px1> <py1> <v1> ...
```

where coordinates are normalized between `0.0` and `1.0`, `v=0` indicates a corner keypoint not in view, and `v=2` indicates a visible keypoint.

**5. Naming Conventions & Dataset Splitting**
* **File Names:** Label files retain the exact same base filename as their image (e.g., `frame_001.jpg` -> `frame_001.txt`)
* **Train/Val/Test Policy:** Split final verified pairs into an 80/10/10 ratio (`train/`, `val/`, `test/`).

## Data Preparation
**1.**
Inside your cloned `auto_data_labeling` repository, create a directory to hold your raw images:
```bash
mkdir -p data/your_dataset_name/raw_imgs
```

**2.**
Place all the raw dataset images you want to annotate (must be .png, .jpg, or .jpeg) into this folder.

## Inferencing
Submit the inference script to the Slurm scheduler:
```bash
sbatch submit_inference.sh --input_dir ./data/your_dataset_name/raw_imgs --output_dir ./data/your_dataset_name/predictions
```
- `--input_dir` is where to pull the images from
- `--output_dir` is where to save the raw_predictions.json file to
- `--visualize` (optional) also saves a mask + corner overlay for each image to `<output_dir>/visualizations/`. Useful for debugging a small run; leave it off for a full run since it adds time and disk.
- `--device` (optional) overrides the device (e.g. `cuda:0`, `cpu`). By default it picks CUDA, then MPS, then CPU.
- `--resume` and `--save_every` are covered below.

The job's time limit is set in `submit_inference.sh` (`#SBATCH --time`, currently 24 hours).

**Checkpointing and resuming:** progress is saved to `raw_predictions.json` every 25 images (change with `--save_every N`). If a job crashes, hits the time limit, or you cancel it with `scancel <jobid>`, rerun the same command with `--resume` and the same `--output_dir`. It skips images already done and continues from there.
```bash
sbatch submit_inference.sh --input_dir ./data/your_dataset_name/raw_imgs --output_dir ./data/your_dataset_name/predictions --resume
```
- An image where nothing was found is saved as `null`, so it still shows up in the review tool and you can add the corners by hand.
- If an image errors (corrupt file, out of memory), the run continues, and the image is listed in `<output_dir>/failed_images.json`. It is not saved to `raw_predictions.json` until a `--resume` run succeeds on it, so it won't appear in review until then.

To see live outputs and track inference progress:
```bash
# Check your job ID
squeue -u $USER

# To see how many jobs are before you
squeue -p ocf-hpc

# Replace 123456 with your actual job ID to view a live inference log
tail -f slurm-123456.out
```

## Human Review

Before you start, double-check that your .venv is activated!

**1.**
Launch the Gradio UI
```bash
bash launch_QA.sh --image_dir ./data/your_dataset_name/raw_imgs --json_path ./data/your_dataset_name/predictions/raw_predictions.json
```
- `--image_dir` is where the raw images are for rendering
- `--json_path` is where the generated predictions are
- `--format {pose,detect,both}` picks which labels are written (default `pose`, which is the original behavior). Use `both` to produce the YOLO-pose labels and the single-class detect labels from the same review.
- `--skip_truncated` skips single-post gates in both exports instead of keeping them (by default they are kept: pose marks the missing corners `v=0`, detect extends the box to the image edge)
- `--detect_dir` changes where the detect labels and `manifest.csv` are written (default `<dataset>/intro_detect`)
- `--images {none,copy,link}` controls whether the detect export also places each image next to its label (default `none`, labels only)
- `--port` sets the port for the review UI (default 7860)
- `--resume` skips images that an earlier session already reviewed (they have a label or a manifest row), so you can stop at any time and pick up where you left off

**2.** Review the annotations:
- Click the **public gradio.live** link generated in your terminal to open the UI in your web browser.
- Review the predicted corners (cyan dots) on the underwater gate frame.
- If a corner is incorrect, select the corresponding radio button (e.g., TL for Top-Left) and click on the image to manually move the point. If the pipeline marked something that is not a gate corner at all, select it and click **Remove Selected Corner** to delete it. The same works for an image where nothing was found: click all four corners.
- The line under the image shows the orientation class the current corners would export as, with the edge ratio. It updates as you move corners.
- Click Accept & Export to save the frame and move to the next image. An image with no corners writes nothing.
- Click **No Gate** only when you have confirmed the image contains no gate. It writes an empty label (a negative example). Never use Accept for this.
- When finished, a completion screen will appear.

**Removing corners:** a removed corner is saved as not visible (`v=0` in the pose label). If you remove one corner, the box uses the other three. If you remove both corners of a post, the image becomes a single-post gate (kept with the box extended to the image edge, or skipped with `--skip_truncated`). If the whole detection is wrong and there is no gate in the image, use **No Gate** instead of removing every corner: accepting with no corners writes nothing and the image is recorded as skipped.

**Working in sessions:** every Accept / No Gate is saved immediately, so you can close the tool at any point. The tool does not remember its place on its own: relaunch with `--resume` to continue from the first image you haven't reviewed. Without `--resume` it starts at the first image again, and accepting an image you already reviewed re-exports the *original* prediction over your correction. Images you never reviewed have no label, so the intro project's `split --labels` leaves them out of the dataset.

## Intro Project Export (YOLO-detect)
 
`--format detect` (or `both`) also writes plain YOLO-detect labels for the intro perception project, derived from the same four corners as the pose labels, so the corners stay the single source of truth. There is one class, `0` = gate, and no orientation anywhere.
 
**Label:** `class cx cy w h`, normalized. The box is the bounding box of the visible corners.
 
**Cut-off gates are kept by default.** If a whole post is missing (both left or both right corners), the gate runs off that side of the image, so the box is extended to that image edge. If both top (or both bottom) corners are missing, the box is extended to the top (or bottom) edge. Add `--skip_truncated` to skip single-post gates instead. If the second post is actually visible in the image, click its corners in the review UI to turn it into a whole gate.
 
**What gets written:**
- Accept & Export on a usable gate: a `<name>.txt` with one box.
- **No Gate**: an empty `<name>.txt`. In YOLO format an empty label means "no object in this image" (a background image), so use it only when you have confirmed there is no gate.
- Accept with no usable corners, or a skipped single post: nothing is written and the reason goes in the manifest. Do not hand these over as empty "no gate" labels, because a gate may be visible in them.
**Output:** `intro_detect/` holds the label files and `manifest.csv`, with one row per reviewed image: `filename`, `status` (`exported` / `skipped` / `no_gate`), `reason`, `truncated`, `n_corners`, and the box (`box_cx`, `box_cy`, `box_w`, `box_h`). Filter on `truncated` to find the cut-off gates, or sort by `box_w` to inspect small or distant ones. Re-reviewing an image replaces its earlier export, so labels never go stale.
 
**Images:** by default only labels are written (`--images none`). `--images copy` (or `link` to symlink instead of copying) also places each image next to its label, which is the layout the intro project's current `split` expects.
 
**Handing the data to the intro project:** either use `--images copy` and put the contents of `intro_detect/` into the intro project's `data/raw/`, or point the intro project's `split` at the images and `intro_detect/` separately once it supports a separate labels folder. Warning: the intro project currently still treats class `0` as `gate_left`, so these labels are only meaningful there after it has been moved to a single class.
## Enrich Export (`--enrich`, experimental)

Collects as much as possible about each gate into one label: the gate box and corners **plus** a box for each part. This is **not a standard YOLO format**, so it is its own mode and never mixes with the pose/detect exports.

```
class x y w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4  left_post(x y w h)  right_post(x y w h)  crossbar(x y w h)
```

One line, 29 numbers (columns 0-28), normalized by the image size. `x y w h` is the box centre and size (like YOLO). The corners are TL, TR, BL, BR with `v = 2` visible or `0` missing. A part that was not found is `0 0 0 0`, so later columns never shift: left post is columns 17-20, right post 21-24, crossbar 25-28. The middle divider is not part of the label (it was dropped; to bring it back, add `"middle_divider"` to `PART_NAMES` in `utils.py`).

```bash
# 1. Inference: same command plus --enrich. Writes raw_predictions.json (unchanged) and raw_parts.json
sbatch submit_inference.sh --input_dir ./data/your_dataset_name/raw_imgs --output_dir ./data/your_dataset_name/predictions --enrich --visualize

# 2. Review: the part boxes are drawn on the image. Writes <dataset>/labels_enrich/ INSTEAD of labels/ and intro_detect/
bash launch_QA.sh --image_dir ./data/your_dataset_name/raw_imgs --json_path ./data/your_dataset_name/predictions/raw_predictions.json --enrich
```

- The posts' boxes come from the same cleaned masks the corners come from. The crossbar needs a **second SAM3 pass** (`"horizontal pole"`) and is only looked for when two posts were found; it must span the posts and sit at their tops (`post_selection.select_crossbar`).
- In the review tool, pick the part (**Part box**), then pick what a click does (**Click action**):
  - **Set corner**: the click moves the selected corner (the normal behaviour).
  - **Move part box**: the click is the new **centre** of the selected part's box. Its size is kept, and it is pushed back inside the image near an edge.
  - **Draw part box (2 clicks)**: click two opposite corners of the box, in any order. This also adds a box that was missing. A white cross marks the first click; **Cancel first corner** (or changing the part or action) throws it away. A box smaller than 4 px is rejected.
  - **Remove Selected Part** drops a wrong box (it is exported as `0 0 0 0`). Deleting both corners of a side also drops that post's box.
- `--parts_path` points at `raw_parts.json` if it is not next to `--json_path`. `--resume` works as usual (it looks in `labels_enrich/`).
- Single-post gates and cut-off gates export the same way as in the other modes (`--skip_truncated` applies).