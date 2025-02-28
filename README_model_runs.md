# Running Reasoning Models with Slurm

This directory contains scripts to run the reasoning model for different model sizes (1B, 7B, and 14B) using Slurm's job array functionality.

## Files

- `run_models_slurm.sh`: The Slurm job array script that submits jobs for each model size.
- `run_model_sizes.py`: A Python script that handles running the reasoning model for a specific model size based on the Slurm array task ID.
- `scripts/reasoning_model_measure_time.py`: The main script that runs the reasoning model and measures time.
- `questions.json`: A collection of questions to test the reasoning model.

## Usage

To run the models for all sizes (1B, 7B, and 14B), simply submit the Slurm job array:

```bash
sbatch run_models_slurm.sh
```

This will:
1. Create a `slurmlogs` directory if it doesn't exist
2. Create a `results` directory if it doesn't exist
3. Submit 3 jobs (one for each model size: 1B, 7B, and 14B)
4. Each job will run the reasoning model with the "natalia_clips" question
5. Output logs will be saved in `slurmlogs/model_size_X.log` where X is the array task ID (0, 1, or 2)

## Customization

- To change the question used, modify the `question_key` variable in `run_model_sizes.py`.
- To adjust the maximum number of tokens generated, modify the `--max_new_tokens` parameter in `run_model_sizes.py`.
- To change other parameters, add them to the `cmd` list in `run_model_sizes.py`.

## Requirements

- Slurm workload manager
- Python 3.6+
- PyTorch
- Transformers library 