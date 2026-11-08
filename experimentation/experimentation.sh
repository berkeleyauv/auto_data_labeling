#!/bin/bash -l
#SBATCH --job-name=SAM3_experiment
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

cd ~/auto_data_labeling
source .venv/bin/activate
python experimentation/experimentation.py "$@"