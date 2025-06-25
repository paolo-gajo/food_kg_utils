#!/bin/bash
#SBATCH -J gnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log
#SBATCH --mem=64g
# SBATCH --gres=gpu:1

# mkdir -p .slurm
nvidia-smi
. .env/bin/activate

python ./src/scrape/scrape.py 