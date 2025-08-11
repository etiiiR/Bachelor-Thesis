#!/bin/bash
#SBATCH --job-name=test_p2mpp_2_views_finetune
#SBATCH --partition=performance
#SBATCH --mem=40G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/test_p2mpp_2_views_finetune_%j.out
#SBATCH --error=logs/test_p2mpp_2_views_finetune_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --requeue

export WANDB_API_KEY=XXXXXXXXXXXXXXX

singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/Pixel2MeshPlusPlus/:/code \
  --pwd /code \
  pixel2meshplusplus_latest.sif \
  python test_p2mpp.py -f cfgs/p2mpp_2_views_finetune.yaml
