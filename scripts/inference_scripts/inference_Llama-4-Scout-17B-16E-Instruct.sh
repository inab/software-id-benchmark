#!/bin/bash

# Project directory
PROJECT_DIR="$HOME/projects/software-observatory/software-id-benchmark"

# Script paths
SCRIPT_PATH="src/cli/make_inference_detailed.py"

# Arguments
MESSAGES_FILE="scripts/evaluation/messages_chat.jsonl"
MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
PROVIDER="together"
RESULTS_FILE="scripts/evaluation/results_Llama-4-Scout-17B-16E-Instruct.jsonl"
RAW_RESULTS_PATH="scripts/evaluation/raw_Llama-4-Scout-17B-16E-Instruct_results"
ENV_FILE=".env"

# Change to the project directory
cd "$PROJECT_DIR" || exit 1 

export PYTHONPATH="$PROJECT_DIR"

echo "Running the test inference with llama-3.1-8b-instruct (HuggingFace - Novita) ..." | tee -a rs-inference-Llama-4-Scout-17B-16E-Instruct.log
python3 "$SCRIPT_PATH" \
    --messages-file "$MESSAGES_FILE" \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --results-file "$RESULTS_FILE" \
    --raw-results-path "$RAW_RESULTS_PATH" 2>&1 | tee -a rs-inference-Llama-4-Scout-17B-16E-Instruct.log