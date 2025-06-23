#!/bin/bash
#SBATCH -J run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log
#SBATCH --mem=64G

mkdir -p .slurm
nvidia-smi
source .env_food_recommender/bin/activate

python3 $1