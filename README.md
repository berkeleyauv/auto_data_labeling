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
├── predictions/      # Raw model outputs (raw_predictions.json)
└── labels/           # Final exported annotations (.txt files)
└── intro_detect/     # Intro-project export: images + labels side by side + manifest.csv (only with --format detect/both)
```

**2. Class IDs**
* YOLO-pose export (`labels/`): `0`: `gate`
* Intro-project export (`intro_detect/`): `0`: `left` (the gate's right side appears closer), `1`: `head_on`, `2`: `right` (the gate's left side appears closer)

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
- `--format {pose,detect,both}` picks which labels are written (default `pose`, which is the original behavior). Use `both` to produce the YOLO-pose labels and the intro-project dataset from the same review.
- `--head_on_ratio`, `--clear_ratio` set the orientation bands for the intro export (see below)
- `--skip_truncated` makes the pose export skip single-post gates (by default they are kept, with the missing corners marked `v=0`)
- `--link_images` symlinks images into `intro_detect/` instead of copying them
- `--detect_dir` changes where the intro dataset is written

**2.** Review the annotations:
- Click the **public gradio.live** link generated in your terminal to open the UI in your web browser.
- Review the predicted corners (cyan dots) on the underwater gate frame.
- If a corner is incorrect, select the corresponding radio button (e.g., TL for Top-Left) and click on the image to manually move the point. The same works for an image where nothing was found: click all four corners.
- The line under the image shows the orientation class the current corners would export as, with the edge ratio. It updates as you move corners.
- Click Accept & Export to save the frame and move to the next image. An image with no corners writes nothing.
- Click **No Gate** only when you have confirmed the image contains no gate. It writes an empty label (a negative example). Never use Accept for this.
- When finished, a completion screen will appear.

## Intro Project Export (YOLO-detect)
 
The intro perception project trains and scores a plain YOLO detect model whose **class is the gate's orientation**. `--format detect` (or `both`) derives that class from the same four corners as the pose labels, so the corners stay the single source of truth.
 
**How the class is chosen:** take the lengths of the left edge (`TL`-`BL`) and right edge (`TR`-`BR`) and compute `ratio = right / left`. The closer post looks taller.
- `ratio` within `[1/head_on_ratio, head_on_ratio]` -> `head_on` (class 1)
- `ratio >= clear_ratio` -> `left` (class 0, right post closer)
- `ratio <= 1/clear_ratio` -> `right` (class 2, left post closer)
- anything in between is **ambiguous and skipped**, as the intro project's guide says to do
**What is exported:** only images with all four corners visible and a clear orientation. Gates with a missing post or corner, and ambiguous ones, are skipped and nothing is written for them. Skipped images must NOT be handed over as empty "no gate" labels, because a gate is visible in them.
 
**Output:** `intro_detect/` is a flat folder of `<name>.<ext>` + `<name>.txt` (`class cx cy w h`, normalized) plus `manifest.csv`, which records for every reviewed image: status (`exported` / `skipped` / `no_gate`), the reason, class, ratio, both edge lengths in pixels, and the thresholds used. Sort it by `ratio` to inspect the images near the band edges. Re-reviewing an image replaces its earlier export, so labels never go stale.
 
**Calibrate the thresholds before trusting them.** The defaults (`1.06` / `1.25`) are placeholders. The same ratio means a different viewing angle at different distances, and lens distortion can make one post look larger. To choose values: hand-label about 100 images as left / head-on / right / unsure, look at how those labels line up against the manifest ratios, and set the bands where your own judgment switches. Review the held-out validation and test images fully rather than trusting the computed class.
 
**Handing the data to the intro project:** copy the contents of `intro_detect/` into the intro project's `data/raw/` and run its `intro-perception-data split`. The folder is already in the layout that command expects.