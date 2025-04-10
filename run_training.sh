#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training for each prompt type and redirect output to log files
echo "Starting training for math_long..."
python train_sv.py --name math_long > logs/math_long.log 2>&1
echo "Completed math_long. Log saved to logs/math_long.log"

echo "Starting training for scheming_think_long..."
python train_sv.py --name scheming_think_long > logs/scheming_think_long.log 2>&1
echo "Completed scheming_think_long. Log saved to logs/scheming_think_long.log"

echo "Starting training for scheming_answer_long..."
python train_sv.py --name scheming_answer_long > logs/scheming_answer_long.log 2>&1
echo "Completed scheming_answer_long. Log saved to logs/scheming_answer_long.log"

echo "Starting training for math_short..."
python train_sv.py --name math_short > logs/math_short.log 2>&1
echo "Completed math_short. Log saved to logs/math_short.log"

echo "Starting training for scheming_think_short..."
python train_sv.py --name scheming_think_short > logs/scheming_think_short.log 2>&1
echo "Completed scheming_think_short. Log saved to logs/scheming_think_short.log"

echo "Starting training for scheming_answer_short..."
python train_sv.py --name scheming_answer_short > logs/scheming_answer_short.log 2>&1
echo "Completed scheming_answer_short. Log saved to logs/scheming_answer_short.log"

echo "All training completed! Logs saved in the logs directory."