#!/bin/bash
#SBATCH --job-name=demo_reconstruct
#SBATCH --partition=performance
#SBATCH --mem=40G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/test_mvp2m_2_views_%j.out
#SBATCH --error=logs/test_mvp2m_2_views_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --requeue

singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/Pixel2MeshPlusPlus/:/code \
  --pwd /code \
  pixel2meshplusplus_latest.sif \
  python inference.py -f cfgs/mvp2m_2_views.yaml
