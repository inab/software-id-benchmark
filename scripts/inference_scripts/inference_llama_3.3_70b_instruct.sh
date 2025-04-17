#!/bin/bash

# Project directory
PROJECT_DIR="$HOME/projects/software-observatory/software-id-benchmark"

# Script paths
SCRIPT_PATH="src/cli/make_inference_detailed.py"

# Arguments
MESSAGES_FILE="scripts/evaluation/messages_chat.jsonl"
MODEL="meta-llama/llama-3.3-70b-instruct"
PROVIDER="openrouter"
RESULTS_FILE="scripts/evaluation/results_llama_3.3_70b_instruct.jsonl"
RAW_RESULTS_PATH="scripts/evaluation/raw_llama_3.3_70b_instruct_results"
ENV_FILE=".env"

# Change to the project directory
cd "$PROJECT_DIR" || exit 1 

export PYTHONPATH="$PROJECT_DIR"

echo "Running the test inference with llama_3.3_70b_instruct (OpenRouter) ..." | tee -a rs-inference-llama-3.3-70b-instruct.log
python3 "$SCRIPT_PATH" \
    --messages-file "$MESSAGES_FILE" \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --results-file "$RESULTS_FILE" \
    --raw-results-path "$RAW_RESULTS_PATH" 2>&1 | tee -a rs-inference-llama-3.3-70b-instruct.log