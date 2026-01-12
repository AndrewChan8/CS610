#!/bin/bash
#SBATCH --job-name=cs610_gpu_test ### job name
#SBATCH --partition=gpu ### gpu partition
#SBATCH --gpus=1 ### 1 GPU
#SBATCH --constraint=a100,gpu-80gb ### A100 80GB
#SBATCH --cpus-per-task=1 ### 1 CPU core
#SBATCH --mem=10G ### 10G CPU memory
#SBATCH --time=00:05:00 ### Job time: HH:MM:SS
#SBATCH --output=output_%j.out ### output file
#SBATCH --error=error_%j.err ### error file
module load cuda
./hello