#!/bin/bash

# Project directory
PROJECT_DIR="$HOME/projects/software-observatory/software-id-benchmark"

# Script paths
SCRIPT_PATH="src/cli/make_inference_detailed.py"

# Arguments
MESSAGES_FILE="scripts/evaluation/messages_chat.jsonl"
MODEL="Llama-4-Maverick-17B-128E-Instruct"
PROVIDER="sambanova"
RESULTS_FILE="scripts/evaluation/results_Llama-4-Maverick-17B-128E-Instruct.jsonl"
RAW_RESULTS_PATH="scripts/evaluation/raw_Llama-4-Maverick-17B-128E-Instruct_results"
ENV_FILE=".env"

# Change to the project directory
cd "$PROJECT_DIR" || exit 1 

export PYTHONPATH="$PROJECT_DIR"

echo "Running the test inference with Llama-4-Maverick-17B-128E-Instruct (HuggingFace - Together) ..." | tee -a rs-inference-Llama-4-Maverick-17B-128E-Instruct.log
python3 "$SCRIPT_PATH" \
    --messages-file "$MESSAGES_FILE" \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --results-file "$RESULTS_FILE" \
    --raw-results-path "$RAW_RESULTS_PATH" 2>&1 | tee -a rs-inference-Llama-4-Maverick-17B-128E-Instruct.log