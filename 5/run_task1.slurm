#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100,gpu-80gb
#SBATCH --time=00:05:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err

module load cuda
./task1