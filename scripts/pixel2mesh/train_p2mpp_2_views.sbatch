#!/bin/bash
#SBATCH --job-name=train_p2mpp_2_views
#SBATCH --partition=performance
#SBATCH --mem=30G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=2-24:00:00
#SBATCH --output=logs/train_p2mpp_2_views_%j.out
#SBATCH --error=logs/train_p2mpp_2_views_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --requeue

export WANDB_API_KEY=XXXXXXXXXXXXXXX

singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/Pixel2MeshPlusPlus/:/code \
  --pwd /code \
  pixel2meshplusplus_latest.sif \
  python train_p2mpp.py -f cfgs/p2mpp_2_views.yaml
