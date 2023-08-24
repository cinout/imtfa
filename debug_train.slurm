#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5


#SBATCH --job-name="debtrain"
#SBATCH --account=punim1623
#SBATCH --time=0-00:20:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=50G

module purge

eval "$(conda shell.bash hook)"
conda activate fct


CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --num-gpus 1 \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025

# TODO: linearly change gpus, IMS_PER_BATCH, BASE_LR

##Log this job's resource usage stats###
my-job-stats -a -n -s
##