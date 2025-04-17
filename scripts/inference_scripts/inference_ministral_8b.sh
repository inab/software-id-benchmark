#!/bin/bash

# Project directory
PROJECT_DIR="$HOME/projects/software-observatory/software-id-benchmark"

# Script paths
SCRIPT_PATH="src/cli/make_inference_detailed.py"

# Arguments
MESSAGES_FILE="scripts/evaluation/messages_chat.jsonl"
MODEL="mistralai/ministral-8b"
PROVIDER="openrouter"
RESULTS_FILE="scripts/evaluation/results_ministral_8b.jsonl"
RAW_RESULTS_PATH="scripts/evaluation/raw_ministral_8b_results"
ENV_FILE=".env"

# Change to the project directory
cd "$PROJECT_DIR" || exit 1 

export PYTHONPATH="$PROJECT_DIR"

echo "Running the test inference with ministral-8b (OpenRouter) ..." | tee -a rs-inference-ministral-8b.log
python3 "$SCRIPT_PATH" \
    --messages-file "$MESSAGES_FILE" \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --results-file "$RESULTS_FILE" \
    --raw-results-path "$RAW_RESULTS_PATH" 2>&1 | tee -a rs-inference-ministral-8b.log