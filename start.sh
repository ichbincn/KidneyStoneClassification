#!/bin/bash
#SBATCH -J kidney
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --reservation=root_114


python train.py --batch_size 2 --epochs 30 --save_epoch 5 --resize_rate 1 --input_path /mntcephfs/lab_data/wangcm/fan/data