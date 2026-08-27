#!/bin/bash
#SBATCH --job-name=SAM3_inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module load cuda python
source venv/bin/activate 

python annotate_dataset.py \
    --input_dir ./data/raw_images \
    --output_dir ./data/predictions