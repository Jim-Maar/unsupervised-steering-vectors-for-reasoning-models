# Training Steering Vectors on Reasoning Models

This directory contains scripts to train steering vectors on reasoning models with different parameter combinations using Slurm's job array functionality.

## Files

- `run_steering_vectors_slurm.sh`: The Slurm job array script that submits jobs for each parameter combination with the 7B model.
- `run_steering_vectors_small_slurm.sh`: The Slurm job array script for experiments with the smaller 1.5B model.
- `run_steering_vectors_experiments.py`: A Python script that handles running the steering vector training for the 7B model.
- `run_steering_vectors_small_model.py`: A Python script for running experiments with the smaller 1.5B model.
- `scripts/reasoning_model_train_steering_vectors.py`: The main script that trains steering vectors on the reasoning model and logs the results.
- `questions.json`: A collection of questions to test the reasoning model.

## Parameter Combinations

The experiments cover the following parameter combinations:

### For 7B Model
- **Model Size**: 7B
- **Normalization Values**: 2.5, 5.0
- **Source and Target Layers**: Both set to None (uses the default layers)
- **Learning Rate**: 0.001
- **Memory Optimization**: 8-bit quantization and buffer offloading (only for 7B and 14B models)

This results in 2 experiments (2 normalization values).

### For 1.5B Model
- **Model Size**: 1.5B
- **Normalization Values**: 2.5, 5.0, 7.5, 10.0, 12.5
- **Source Layers**: 3, 4
- **Target Layers**: 6, 7
- **Learning Rate**: 0.01
- **Memory Optimization**: None (not needed for smaller model)

This results in 20 experiments (5 normalization values × 2 source layers × 2 target layers).

## Usage

To run the 7B model experiments:

```bash
sbatch run_steering_vectors_slurm.sh
```

To run the 1.5B model experiments:

```bash
sbatch run_steering_vectors_small_slurm.sh
```

These will:
1. Create a `slurmlogs` directory if it doesn't exist
2. Create the appropriate results directories
3. Submit jobs for each parameter combination
4. Each job will train steering vectors on the reasoning model with the "cube_probability_3" question
5. Output logs will be saved in the appropriate log files
6. Results will be saved in the appropriate directories based on model size and parameters

## Memory Optimization

To handle the large 7B and 14B models, the script uses the following memory optimization techniques:

- **8-bit Quantization**: Reduces the precision of model weights to save memory
- **Buffer Offloading**: Offloads some buffers to CPU to reduce GPU memory usage
- **Half Precision**: Uses torch.float16 for model weights

These optimizations allow the larger models to fit in GPU memory while still maintaining reasonable performance. For the smaller 1.5B model, these optimizations are not applied as they are unnecessary and might impact performance.

## Customization

- To change the question used, modify the `question_key` variable in the appropriate experiment script.
- To adjust the maximum number of tokens generated, modify the `--max_new_tokens` parameter.
- To change the number of training steps, modify the `--num_steps` parameter.
- To change the number of steering vectors, modify the `--num_vectors` parameter.
- To add more parameter combinations, update the parameter lists and adjust the `--array` parameter in the Slurm script accordingly.

## Requirements

- Slurm workload manager
- Python 3.6+
- PyTorch
- Transformers library
- bitsandbytes library (for quantization with larger models) 