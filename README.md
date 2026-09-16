# auto_data_labeling

Automated segmentation and bounding box labeling pipeline using SAM3. This  automatically generates normalized YOLO-pose keypoint data for underwater gate detection.

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
```

**2. Class IDs**
* `0`: `gate`

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

**2.** Review the annotations:
- Click the **public gradio.live** link generated in your terminal to open the UI in your web browser.
- Review the predicted corners (cyan dots) on the underwater gate frame.
- If a corner is incorrect, select the corresponding radio button (e.g., TL for Top-Left) and click on the image to manually move the point.
- Click Accept & Export to save the frame and move to the next image.
- When finished, a completion screen will appear.