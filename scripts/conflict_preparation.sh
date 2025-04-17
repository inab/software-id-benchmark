#!/bin/bash

# Project directory
PROJECT_DIR="$HOME/projects/software-observatory/software-id-benchmark"

# File paths
SCRIPT_PATH="src/cli/prepare_conflicts.py"
GROUPED_ENTRIES_FILE="data/grouped_entries.json"
DISCONNECTED_ENTRIES_FILE="data/disconnected_entries.json"
EVALUATION_CASES_CSV="scripts/evaluation/software_disambiguation_evaluation_conflicts.csv"
PROMPT_TYPE="chat"
MESSAGES_FILE="scripts/evaluation/messages_chat.jsonl"
ENV_FILE=".env"

# Change to the project directory
cd "$PROJECT_DIR" || exit 1 

export PYTHONPATH="$PROJECT_DIR"

# Run the Python script with the provided arguments
echo "1. Running conflict preparation with chat style prompt for evaluation cases..." | tee -a rs-eval-message-prep.log


python3 "$SCRIPT_PATH" \
    --disconnected-entries-file "$DISCONNECTED_ENTRIES_FILE" \
    --grouped-entries-file "$GROUPED_ENTRIES_FILE" \
    --evaluation-cases-csv "$EVALUATION_CASES_CSV" \
    --messages-file "$MESSAGES_FILE" \
    --prompt-type "$PROMPT_TYPE" 2>&1 | tee -a rs-eval-message-prep.log
