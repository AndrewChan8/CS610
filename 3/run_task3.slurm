#!/bin/bash
#SBATCH --job-name=task03
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100,gpu-80gb
#SBATCH --time=00:05:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err

module load cuda
./task03 1
mv output.ppm output_radius_1.ppm

./task03 2
mv output.ppm output_radius_2.ppm

./task03 4
mv output.ppm output_radius_4.ppm

./task03 8
mv output.ppm output_radius_8.ppm