#!/bin/bash
#SBATCH -A REPLACE_ME
#SBATCH -p REPLACE_ME
#SBATCH --qos=REPLACE_ME
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 1                # n node
#SBATCH --ntasks-per-node=4 # tasks out of 32
#SBATCH --gres=gpu:4        # gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB
#SBATCH --job-name=cuda_test_4gpu
#SBATCH --error=cuda_test_4gpu.err
#SBATCH --output=cuda_test_4gpu.out
#SBATCH --cpus-per-task=4

export OMP_PROC_BIND=spread
export OMP_PLACES=threads 

srun example