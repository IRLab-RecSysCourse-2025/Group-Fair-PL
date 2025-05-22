#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=groupfairpl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:59:00
#SBATCH --output=/scratch-shared/%u/logs/%x-%A.out

date

# Definitions
export HF_HOME="/scratch-shared/$USER"
export HF_TOKEN=
export WANDB_API_KEY=
export GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

# make sure the correct modules are used and that the virtual environment is active
# prerequisite: the virtual environment must be created with the same python version as the module
PROJECT_ROOT=$(pwd)
source $PROJECT_ROOT/scripts/slurm_setup.sh
setup $PROJECT_ROOT
cd $PROJECT_ROOT

# Run the training script
python -m src.main --file config/config_German.jsonc --loss groupfairpl --postprocess_algorithms none,GDL23,GAK19 --run_no 1 --bias 9