#!/bin/bash

# Project directory
PROJECT_DIR="$HOME/projects/software-observatory/software-id-benchmark"

# Script paths
SCRIPT_PATH="src/cli/make_inference_detailed.py"

# Arguments
MESSAGES_FILE="scripts/evaluation/messages_chat.jsonl"
MODEL="mistralai/mistral-7b-instruct-v0.3"
PROVIDER="together"
RESULTS_FILE="scripts/evaluation/results_mistral_7b_instruct_v0.3.jsonl"
RAW_RESULTS_PATH="scripts/evaluation/raw_mistral_7b_instruct_v0.3_results"
ENV_FILE=".env"

# Change to the project directory
cd "$PROJECT_DIR" || exit 1 

export PYTHONPATH="$PROJECT_DIR"

echo "Running the test inference with mistral_7b_instruct (HuggingFace) ..." | tee -a rs-inference-mistral-7b-instruct_v0.3.log
python3 "$SCRIPT_PATH" \
    --messages-file "$MESSAGES_FILE" \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --results-file "$RESULTS_FILE" \
    --raw-results-path "$RAW_RESULTS_PATH" 2>&1 | tee -a rs-inference-mistral-7b-instruct_v0.3.log