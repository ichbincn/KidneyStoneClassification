#!/bin/bash
#SBATCH -J kidney
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --reservation=root_114


python train.py --epochs 30 --input_size "128,128,128" --batch_size 2 --input_path /mntcephfs/lab_data/wangcm/fan/data