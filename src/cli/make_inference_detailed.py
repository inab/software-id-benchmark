""" 
The command-line interface for the inferences (keeping raw responses and stats, for benchmarking)
""" 

import argparse
import logging
from dotenv import load_dotenv 

logger = logging.getLogger("rs-etl-pipeline") 

def main():
    parser = argparse.ArgumentParser(
        description="""Make inferences for benchmarking. 
        The providers supported are HuggingFace (https://huggingface.co/) and OpenRouter (https://openrouter.ai/).  
        Each of this providers has its own API key.
        The API key for HuggingFace is set in the environment variable HUGGINGFACE_API_KEY.
        The API key for OpenRouter is set in the environment variable OPENROUTER_API_KEY. 
        The way the prompt is passed to the model is different for each provider.
        The prompt for HuggingFace is passed as a string (refered to as "flattened" in this codebase).
        The prompt for OpenRouter is passed as a list of dictionaries.
        """
    )
    parser.add_argument(
        "--messages-file", "-m",
        help=("Path to the file containing messages for inference."),
        type=str,
        dest="messages_file",
    )
    parser.add_argument(
        "--model", "-M",
        help=("Model to use for inference."),
        type=str,
        dest="model",
    )
    parser.add_argument(
        "--provider", "-p",
        help=("Provider to use for inference (huggingface | openrouter)."),
        type=str,
        dest="provider",
    )
    parser.add_argument(
        "--results-file", "-r",
        help=("Path to the file where the results will be written. Format: JSONL"),
        type=str,
        dest="results_file",
    )
    parser.add_argument(
        "--raw-results-path", "-R",
        help=("Path to the file where the raw results and stats of each inference will be written."),
        type=str,
        dest="raw_results_path",
    )

    parser.add_argument(
        "--env-file", "-e",
        help=("File containing environment variables to be set before running. The API key for the provider is set in the environment variable."),
        default=".env",
    )

    args = parser.parse_args()

    # Load the environment variables ------------------------------------------
    logger.debug(f"Env file: {args.env_file}")
    load_dotenv(args.env_file)

    # import the function to make inferences
    from src.application.use_cases.make_inferences import make_inferences_model
    # call the function to make inferences
    logger.info(f"Making inferences for model {args.model} with provider {args.provider}")
    logger.info(f"Messages file: {args.messages_file}")
    logger.info(f"Results file: {args.results_file}")
    logger.info(f"Raw results path: {args.raw_results_path}")
    logger.info(f"Provider: {args.provider}")
    logger.info(f"🚀 Starting inferences...")
    make_inferences_model(args.messages_file, args.model, args.results_file, args.raw_results_path, args.provider)
    logger.info("Inferences completed.")

if __name__ == "__main__":
    main()