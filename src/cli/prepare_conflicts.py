# Args
# disconnected_entries_file
#    grouped_entries_file
#    evaluation_cases_csv
#    messages_file
#    prompt_type
"""
The command-line interface for the disambiguation step of the integration
"""

import argparse
import logging
import asyncio
from dotenv import load_dotenv

logger = logging.getLogger("rs-etl-pipeline")

async def main():
    parser = argparse.ArgumentParser(
        description="""Prepare messages for inference."""
    )
    parser.add_argument(
        "--grouped-entries-file", "-g",
        help=("Path to the file containing grouped entries. Default is 'data/grouped.json'."),
        type=str,
        dest="grouped_entries_file",
        default="data/grouped.json",
    )

    parser.add_argument(
        "--evaluation-cases-csv", "-c",
        help=("Path to the CSV file containing evaluation cases. Default is 'scripts/data/evaluation/software_disambiguation_evaluation_conflicts.csv'."),
        type=str,
        dest="evaluation_cases_csv",
        default="scripts/data/evaluation/software_disambiguation_evaluation_conflicts.csv",
    )

    parser.add_argument(
        "--messages-file", "-m",
        help=("Path to the file where the messages for inference will be written. Default is 'scripts/data/evaluation/messages.json'."),
        type=str,
        dest="messages_file",
        default="scripts/data/evaluation/messages.json",
    )

    parser.add_argument(
        "--prompt-type", "-p",
        help=("Type of prompt to use. Default is 'chat'."),
        type=str,
        dest="prompt_type",
        default="chat",
    )

    parser.add_argument(
        "--env-file", "-e",
        help=("File containing environment variables to be set before running "),
        default=".env",
    )

    args = parser.parse_args()
    # Load the environment variables ------------------------------------------
    logger.debug(f"Env file: {args.env_file}")
    load_dotenv(args.env_file)

    from src.application.use_cases.message_preparation import prepare_messages

    await prepare_messages(args.grouped_entries_file, args.evaluation_cases_csv, args.messages_file, args.prompt_type)


if __name__ == "__main__":
    asyncio.run(main())