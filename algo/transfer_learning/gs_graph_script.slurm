#!/bin/bash

#SBATCH --job-name=gs_graph_sep_1

##ARGO
##SBATCH --partition=gpuq --gres=gpu:1 --constraint=gpu-v100

##HOPPER
#SBATCH --qos=gpu --partition=gpuq --gres=gpu:A100.40gb:1

##SBATCH --output=/scratch/%u/slurm_logs/%x-%j-%N.out
#SBATCH --output=%x-%j-%N.out
##SBATCH --error=/scratch/%u/slurm_logs/%x-%j-%N.err
#SBATCH --error=%x-%j-%N.err

#SBATCH --mem=80G
#SBATCH --time=5-00:00:00

#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --export=ALL

## env
source env/bin/activate
python grid_style_graph.py
