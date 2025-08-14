#!/bin/bash
#SBATCH --job-name=batch_inference
#SBATCH --partition=performance
#SBATCH --mem=40G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/batch_inference_%j.out
#SBATCH --error=logs/batch_inference_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --requeue

export WANDB_API_KEY=XXXXXXXXXXXXXXX

singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/Pixel2MeshPlusPlus/:/code \
  --pwd /code \
  pixel2meshplusplus_latest.sif \
  python inference.py
