#!/bin/bash
#SBATCH --job-name=pixel2mesh_array
#SBATCH --partition=performance
#SBATCH --mem=50G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=6-24:00:00
#SBATCH --output=logs/pixel2mesh_array_%A_%a.out
#SBATCH --error=logs/pixel2mesh_array_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-19%3  # 30 jobs, max 3 at once

# List all train jobs (edit to add/remove as needed)
JOBS=(
    "test_mvp2m_2_views.sbatch"
"train_mvp2m_2_views_finetune.sbatch"
"train_mvp2m_2_views_freeze.sbatch"
"test_mvp2m_2_views_mean_prior.sbatch"
"test_mvp2m_2_views_spherical_prior.sbatch"
"test_mvp2m_3_views.sbatch"
"test_mvp2m_4_views.sbatch"
"test_mvp2m_5_views.sbatch"
"test_mvp2m_6_views.sbatch"
)

cd /home2/etienne.roulet/slurm/pixel2mesh/
sbatch "${JOBS[$SLURM_ARRAY_TASK_ID]}"
