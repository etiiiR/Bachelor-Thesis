#!/bin/bash
#SBATCH --job-name=train_mvp2m_none_augmentation
#SBATCH --partition=performance
#SBATCH --mem=70G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=2-24:00:00
#SBATCH --output=logs/train_mvp2m_none_augmentation_%j.out
#SBATCH --error=logs/train_mvp2m_none_augmentation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --requeue

export WANDB_API_KEY=XXXXXXXXXXXXXXX

singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/Pixel2MeshPlusPlus/:/code \
  --pwd /code \
  pixel2meshplusplus_latest.sif \
  bash -c "python generate_mvp2m_intermediate.py -f cfgs/mvp2m_none_augmentation.yaml"
