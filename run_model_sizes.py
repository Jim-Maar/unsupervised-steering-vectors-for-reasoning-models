import os
import sys
import subprocess
import json
import shlex

# Get the Slurm array task ID (will be 0, 1, or 2 for the three model sizes)
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

# Map task ID to model size
model_sizes = [1, 7, 14]
model_size = model_sizes[task_id]

# Load a question from the questions.json file
with open('questions.json', 'r') as f:
    questions = json.load(f)

# Use the first question (natalia_clips) for simplicity
question_key = "natalia_clips"
question = questions[question_key]

# Properly escape the question for command line
escaped_question = shlex.quote(question)

# Set up command to run the reasoning model script
cmd = [
    "python", 
    "scripts/reasoning_model_measure_time.py",
    f"--model_size={model_size}",
    f"--question={escaped_question}",
    "--max_new_tokens=10",
    f"--save_dir=results/model_{model_size}B"
]

# Print the command being executed
print(f"Running model size: {model_size}B")
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