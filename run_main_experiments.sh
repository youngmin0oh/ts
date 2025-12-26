#!/bin/bash

# Script to reproduce Main Experiments for delta-Adapter (Ada-X+Y)
# Reference: Tables 1, 2, and 3 in the PDF.

# Note: This script assumes running on ETT datasets which are available in ./datasets/
# The PDF mentions delta=0.01 for ETT datasets and delta=0.1 for others (Traffic, Weather, etc.)

set -e

# Define paths
ROOT_DIR=$(pwd)
DATASET_DIR="$ROOT_DIR/datasets/"
ADAPTER_DIR="$ROOT_DIR/Adapter-X+Y"

# Navigate to Adapter directory
cd "$ADAPTER_DIR"

# Experiment Settings
# Datasets available locally
datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2")

# Models to evaluate (selected from Table 2 and 3)
models=("iTransformer" "Autoformer" "FreTS" "FourierGNN")

# Delta parameter for ETT datasets (from PDF Section 5)
delta=0.01

# Sequence and Prediction lengths
seq_len=96
pred_len=96

# Training Epochs (Set to 1 for demonstration/speed, Paper likely uses 10)
train_epochs=1

echo "Starting Main Experiments for delta-Adapter (Ada-X+Y)"
echo "====================================================="

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running Experiment: Model=$model, Dataset=$dataset, Delta=$delta"
    echo "----------------------------------------------------------------"
    
    # We use run_xy_add.py as it implements the Additive Ada-X+Y which is the primary method.
    # We enable is_training to first train the base model (simulating pre-training) 
    # and then train the adapter.
    
    python -u run_xy_add.py \
      --is_training 1 \
      --root_path "$DATASET_DIR" \
      --data_path "${dataset}.csv" \
      --model_id "${dataset}_${seq_len}_${pred_len}" \
      --model "$model" \
      --data "$dataset" \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --des 'Exp' \
      --itr 1 \
      --delta $delta \
      --learning_rate 0.0001 \
      --train_epochs $train_epochs \
      --batch_size 32
      
    echo "Finished Experiment for $model on $dataset"
    echo ""
  done
done

echo "====================================================="
echo "All Experiments Completed."
echo "Results (logs) are displayed above and saved in $ADAPTER_DIR/results/"
