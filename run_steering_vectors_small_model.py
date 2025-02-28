import os
import sys
import subprocess
import json
import shlex
import argparse
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='Run steering vector experiments with small model')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID for Slurm array job')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the Slurm array task ID (or use the provided task_id argument)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', args.task_id))
    
    # Fixed parameters
    model_size = 1  # 1.5B model
    learning_rate = 0.001  # Fixed learning rate for smaller model
    
    # Variable parameters
    # normalization_values = [2.5, 5.0, 7.5, 10.0, 12.5]
    # source_layers = [3, 4]
    # target_layers = [6, 7]
    normalization_values = [None]
    source_layers = [None]
    target_layers = [None]
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(normalization_values, source_layers, target_layers))
    
    # Calculate which parameter combination to use based on task_id
    param_idx = task_id % len(param_combinations)
    normalization, source_layer, target_layer = param_combinations[param_idx]
    
    # Load a question from the questions.json file
    with open('questions.json', 'r') as f:
        questions = json.load(f)
    
    # Use the cube_probability_3 question
    question_key = "betty_wallet"
    question = questions[question_key]
    
    # Properly escape the question for command line
    escaped_question = shlex.quote(question)
    
    # Create results directory if it doesn't exist
    os.makedirs('results/steering_vectors/model_1.5B', exist_ok=True)
    
    # Format source and target layer for filename
    src_str = str(source_layer)
    tgt_str = str(target_layer)
    
    # Set up save directory
    save_dir = f"results/steering_vectors/model_1.5B/norm_{normalization}_src_{src_str}_tgt_{tgt_str}_lr_{learning_rate}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up command to run the reasoning model script
    cmd = [
        "python", 
        "scripts/reasoning_model_train_steering_vectors.py",
        f"--model_size={model_size}",
        f"--question={escaped_question}",
        f"--normalization={normalization}",
        f"--source_layer={source_layer}",
        f"--target_layer={target_layer}",
        f"--learning_rate={learning_rate}",
        f"--save_dir={save_dir}",
        f"--num_steps=400",
        f"--num_vectors=100",
        f"--max_new_tokens=200",
    ]
    
    # Print the command being executed
    print(f"Running model size: {model_size}B (1.5B)")
    print(f"Normalization: {normalization}")
    print(f"Source layer: {source_layer}")
    print(f"Target layer: {target_layer}")
    print(f"Learning rate: {learning_rate}")
    print(f"Question: {question}")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output
    print("STDOUT:")
    print(result.stdout)
    
    print("STDERR:")
    print(result.stderr)
    
    # Exit with the same code as the subprocess
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 