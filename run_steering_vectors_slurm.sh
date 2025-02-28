#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --job-name=steering_vectors
#SBATCH --partition sorcery
#SBATCH --output=slurmlogs/steering_vectors_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=1-0:0:0
#SBATCH --constraint=ARCH:X86
#SBATCH --array=0-1

# Debug information
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la
echo "Python version:"
python --version
echo "PYTHONPATH: $PYTHONPATH"

# Create the slurmlogs directory if it doesn't exist
mkdir -p slurmlogs

# Create the results directory if it doesn't exist
mkdir -p results/steering_vectors/model_7B

# Check if src directory exists
echo "Checking if src directory exists:"
if [ -d "src" ]; then
    echo "src directory exists"
    echo "Contents of src directory:"
    ls -la src
else
    echo "src directory does not exist"
fi

# Run the Python script that handles the job array
python -u run_steering_vectors_experiments.py 