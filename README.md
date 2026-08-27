# auto_data_labeling

Automated segmentation and bounding box labeling pipeline using SAM3. This  automatically generates normalized YOLO-pose keypoint data for underwater gate detection.

## Prerequisites
1. **Compute Environment:** 
This pipeline supports CUDA, MPS, and CPU, but running it on a GPU cluster is highly recommended for optimal inferencing speeds

2. **Hugging Face Authentication:** 
You will need a valid Hugging Face access token to download the model weights. Run the following command in your terminal and follow the prompts to paste your token: 
```bash 
huggingface-cli login
```

## Installation
**1.**
Install `uv` once per machine:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(or `brew install uv` on macOS).*

**2.**
Clone this auto-labeling repository and navigate into it:
```bash
git clone https://github.com/berkeleyauv/auto_data_labeling.git
cd auto_data_labeling
```

**3.**
Create the virtual environment pinning Python 3.11, which creates a `.venv/` directory:
```bash
uv venv --python 3.11
source .venv/bin/activate
```

**4.**
Install the necessary PyTorch and computer vision libraries from the requirements file:
```bash
uv pip install -r requirements.txt
```

## Data Preparation
**1.**
Inside your cloned `auto_data_labeling` repository, create a directory to hold your raw images:
```bash
mkdir -p data/raw_images
```

**2.**
Place all the raw dataset images you want to annotate (must be .png, .jpg, or .jpeg) into this folder.

## Inferencing
Submit the inference script to the Slurm scheduler:
```bash
sbatch submit_inference.sh
```
- `--input_dir` is where to pull the images from
- `--output_dir` is where to save the raw_predictions.json file to

To see live outputs and track inference progress:
```bash
# Check your job ID
squeue -u $USER

# Replace 123456 with your actual job ID
tail -f slurm-123456.out
```

## Human Review
**1.**
Launch the Gradio UI
```base
bash launch_QA.sh
```
- `--json_path` is where the generated predictions are
- `--image_dir` is where the raw images are for rendering

**2.** Review the annotations:
- Click the public gradio.live link generated in your terminal to open the UI in your web browser.
- Review the predicted corners (cyan dots) on the underwater gate frame.
- If a corner is incorrect, select the corresponding radio button (e.g., TL for Top-Left) and click on the image to manually move the point.
- Click Accept & Export to save the frame and move to the next image.
- When finished, a completion screen will appear.

## Results
- A ./data/labels directory will have been created
- Each image has a corresponding .txt file containing the normalized keypoints and bounding boxes of the gate