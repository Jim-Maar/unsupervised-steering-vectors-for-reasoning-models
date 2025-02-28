import os
import sys
import subprocess
import json
import shlex
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run steering vector experiments')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID for Slurm array job')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the Slurm array task ID (or use the provided task_id argument)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', args.task_id))
    
    # Fixed parameters
    model_size = 7
    source_layer = None
    target_layer = None
    learning_rate = 0.001
    
    # Variable parameters
    normalization_values = [2.5, 5.0]
    
    # Calculate which normalization value to use based on task_id
    norm_idx = task_id % len(normalization_values)
    normalization = normalization_values[norm_idx]
    
    # Load a question from the questions.json file
    with open('questions.json', 'r') as f:
        questions = json.load(f)
    
    # Use the cube_probability_3 question
    question_key = "cube_probability_3"
    question = questions[question_key]
    
    # Properly escape the question for command line
    escaped_question = shlex.quote(question)
    
    # Create results directory if it doesn't exist
    os.makedirs('results/steering_vectors/model_7B', exist_ok=True)
    
    # Format source and target layer for filename
    src_str = "None" if source_layer is None else str(source_layer)
    tgt_str = "None" if target_layer is None else str(target_layer)
    
    # Set up save directory
    save_dir = f"results/steering_vectors/model_7B/norm_{normalization}_src_{src_str}_tgt_{tgt_str}_lr_{learning_rate}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up command to run the reasoning model script
    cmd = [
        "python", 
        "scripts/reasoning_model_train_steering_vectors.py",
        f"--model_size={model_size}",
        f"--question={escaped_question}",
        f"--normalization={normalization}",
        f"--learning_rate={learning_rate}",
        f"--save_dir={save_dir}",
        f"--num_steps=400",
        f"--num_vectors=4",
        f"--max_new_tokens=200",
    ]
    
    # Add memory-saving options only for larger models (7B and 14B)
    if model_size >= 7:
        cmd.extend([
            "--use_8bit",
            "--offload_buffers"
        ])
    
    # Add source_layer and target_layer if they are not None
    if source_layer is not None:
        cmd.append(f"--source_layer={source_layer}")
    if target_layer is not None:
        cmd.append(f"--target_layer={target_layer}")
    
    # Print the command being executed
    print(f"Running model size: {model_size}B")
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