#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --job-name=reasoning_models
#SBATCH --partition sorcery
#SBATCH --output=slurmlogs/model_size_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=1-0:0:0
#SBATCH --constraint=ARCH:X86
#SBATCH --array=0-2

# Create the slurmlogs directory if it doesn't exist
mkdir -p slurmlogs

# Create the results directory if it doesn't exist
mkdir -p results

# Run the Python script that handles the job array
python -u run_model_sizes.py 