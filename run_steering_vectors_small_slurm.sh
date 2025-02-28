#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --job-name=steering_small
#SBATCH --partition sorcery
#SBATCH --output=slurmlogs/steering_small_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=1-0:0:0
#SBATCH --constraint=ARCH:X86
#SBATCH --array=0

# Debug information
echo "Current directory: $(pwd)"
echo "Python version:"
python --version
echo "PYTHONPATH: $PYTHONPATH"

# Create the slurmlogs directory if it doesn't exist
mkdir -p slurmlogs

# Create the results directory if it doesn't exist
mkdir -p results/steering_vectors/model_1.5B

# Check if src directory exists
echo "Checking if src directory exists:"
if [ -d "src" ]; then
    echo "src directory exists"
else
    echo "src directory does not exist"
    exit 1
fi

# Print parameter combinations for reference
echo "Parameter combinations:"
echo "Normalization values: 2.5, 5.0, 7.5, 10.0, 12.5"
echo "Source layers: 3, 4"
echo "Target layers: 6, 7"
echo "Total combinations: 20"

# Run the Python script that handles the job array
python -u run_steering_vectors_small_model.py 