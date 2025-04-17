import logging
import json
import time
from src.application.services.benchmarking import parse_messages_file
from src.application.services.disambiguation import load_solved_conflict_keys, parse_result, query_openrouter, query_huggingface_new
REQUESTS_PER_MINUTE = 20
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

def save_raw_results(result, raw_result_path):
    """
    Save the raw result to a file.
    """
    with open(raw_result_path, 'w') as f:
        json.dump(result, f)
        f.write('\n')
    logging.info(f"Raw result saved to {raw_result_path}")

def save_inference_stats(meta, meta_file_path):
    """
    Save the inference statistics to a file.
    """
    with open(meta_file_path, 'a') as f:
        json.dump(meta, f)
        f.write('\n')
    
    logging.info(f"Metadata saved to {meta_file_path}")

def save_result(key, parsed_result, output_file_path):
    with open(output_file_path, 'a') as f:
                json.dump({key: parsed_result}, f)
                f.write('\n')


def make_inferences(messages_dict, model, results_file_path, raw_results_path, provider):
    '''
    For benchmarking of the disambiguation process.
    This maybe can be moved to a use_case
    '''
    logging.info(f"Number of cases to process: {len(messages_dict)}")

    solved_conflicts_keys = load_solved_conflict_keys(results_file_path)
    count = 0
    for key in messages_dict:
        if key in solved_conflicts_keys:
            logging.info(f"Skipping conflict {key}, already solved.")
            continue


        messages = messages_dict[key]

        if provider == "openrouter":
            time.sleep(DELAY_BETWEEN_REQUESTS)

        logging.info(f"Making inference for conflict {count} - {key} ")
        try:
            start_time = time.time()
            if provider == "openrouter":
                result, meta = query_openrouter(messages, model=model)
            else:
                result, meta = query_huggingface_new(messages, model=model, provider=provider)
            end_time = time.time()
            total_time = end_time - start_time

            key_transformed = key.replace(" ", "_").replace("/", "_")

            ## Save the raw result to a file
            raw_result_path = f"{raw_results_path}/raw_results_{key_transformed}.json"
            save_raw_results(result, raw_result_path)

            ## Parse and save metadata
            meta_file_path = f"{raw_results_path}/meta_{key_transformed}.json"

            meta['total_time'] = total_time
            meta['key'] = key
            save_inference_stats(meta, meta_file_path)


            ## Parse the result
            parsed_result  = parse_result(result)
            save_result(key, parsed_result, results_file_path)
            
            logging.info(f"Results for conflict {key} written to file.")
            logging.info("----------------------------")

            count += 1    
        
        except Exception as e:
            logging.error(f"Error processing conflict {key}: {e}")
            raise


    
    logging.info("All inferences completed.")


def make_inferences_model(messages_file_path, model, results_file_path, raw_results_path, provider):
    """
    Make inferences for a given set of messages with a given model.
    """
    messages_dict = parse_messages_file(messages_file_path)
    make_inferences(messages_dict, model, results_file_path, raw_results_path, provider)
    return



if __name__ == "__main__":
    messages_file = 'scripts/data/evaluation/messages.json'
    model = ''
    output_file_path = 'scripts/data/evaluation/benchmarking_results_model.json'
    raw_results_path = 'scripts/data/evaluation/raw_results'


    # 🤔 Measure time for each inference??

    # Make inferences for each message
    messages = parse_messages_file(messages_file)
    logging.info(f"Making inferences with model {model}")
    make_inferences_model(messages, model, output_file_path)

