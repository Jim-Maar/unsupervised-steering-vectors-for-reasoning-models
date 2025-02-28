#!/bin/bash

# Debug information
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la
echo "Python version:"
python --version
echo "PYTHONPATH: $PYTHONPATH"

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

# Run the experiments for each normalization value
echo "Running experiment with normalization=2.5"
python -u run_steering_vectors_experiments.py --task_id=0

echo "Running experiment with normalization=5.0"
python -u run_steering_vectors_experiments.py --task_id=1

echo "All experiments completed!" 