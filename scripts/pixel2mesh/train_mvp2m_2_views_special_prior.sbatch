#!/bin/bash
#SBATCH --job-name=train_mvp2m_2_views_special_prior
#SBATCH --partition=performance
#SBATCH --mem=70G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=2-24:00:00
#SBATCH --output=logs/train_mvp2m_2_views_special_prior_%j.out
#SBATCH --error=logs/train_mvp2m_2_views_special_prior_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --requeue

export WANDB_API_KEY=XXXXXXXXXXXXXXX

singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/Pixel2MeshPlusPlus/:/code \
  --pwd /code \
  pixel2meshplusplus_latest.sif \
  python train_mvp2m.py -f cfgs/mvp2m_2_views_special_prior.yaml
