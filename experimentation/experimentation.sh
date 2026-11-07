#!/bin/bash -l
#SBATCH --job-name=SAM3_experiment
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
python experimentation/experimentation.py "$@"