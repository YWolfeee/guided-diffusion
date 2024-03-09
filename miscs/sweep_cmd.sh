#!/bin/bash
#SBATCH -A atlas
#SBATCH --partition=atlas
#SBATCH --gres=gpu:a4000:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=atlas23

source ~/.bashrc
conda init
conda activate torch
conda env list

nohup python -u miscs/run.py --gpu_id 0 --mode_id 4 --label_id 0 > logs/040.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 1 --mode_id 4 --label_id 1 > logs/141.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 2 --mode_id 4 --label_id 2 > logs/242.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 3 --mode_id 5 --label_id 0 > logs/350.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 4 --mode_id 5 --label_id 1 > logs/451.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 5 --mode_id 5 --label_id 2 > logs/552.log 2>&1 & 
wait

echo "Finished all"

# nohup python -u miscs/run.py --gpu_id 0 --mode_id 2 --label_id 0 > logs/020.log 2>&1 & 
# nohup python -u miscs/run.py --gpu_id 1 --mode_id 2 --label_id 1 > logs/121.log 2>&1 & 
# nohup python -u miscs/run.py --gpu_id 2 --mode_id 2 --label_id 2 > logs/222.log 2>&1 & 
# nohup python -u miscs/run.py --gpu_id 3 --mode_id 3 --label_id 0 > logs/330.log 2>&1 & 
# nohup python -u miscs/run.py --gpu_id 4 --mode_id 3 --label_id 1 > logs/431.log 2>&1 & 
# nohup python -u miscs/run.py --gpu_id 5 --mode_id 3 --label_id 2 > logs/532.log 2>&1 & 