
import json
import logging 
from src.application.services.disambiguation import build_full_conflict, write_to_results_file, build_chat_messages_with_disconnected, load_templates_from_folder
PROMPT_TEMPLATES = load_templates_from_folder("src/application/services/integration/benchmarking")

# --------------------------------
# Preparing messages for inference
# --------------------------------

def flatten_messages_to_prompt(messages, model_name="default"):
    """
    Convert OpenAI-style chat messages into a single prompt string
    suitable for instruction-tuned models (version2 style).
    """
    # Define simple role tags (can be adjusted for model-specific formatting)
    role_tags = {
        "system": "### System",
        "user": "### User",
        "assistant": "### Assistant"
    }

    prompt_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if role in role_tags:
            prompt_parts.append(f"{role_tags[role]}\n{content}\n")
        else:
            prompt_parts.append(f"### {role.capitalize()}\n{content}\n")

    # Add final assistant marker (helps some models know when to answer)
    prompt_parts.append("### Assistant\n")

    return "\n".join(prompt_parts).strip()

def build_prompt_benchmarking(data_dict, prompt_type):
    """
    Build the prompt for benchmarking. The prompt for the benchmarking is slightly different
    from the one used in the disambiguation process. The benchmarking prompt asks for more details
    about the disambiguation process, such as the key features behind the decision.
    """

    disconnected_preamble = "The first software metadata_entry"
    remaining_preamble = "The second software metadata_entry"


    if prompt_type == "chat":
        template = PROMPT_TEMPLATES["prompt_benchmarking_chat_style"]
        instruction_prompt = template.render()

    
        if len(data_dict['disconnected']) > 1 and len(data_dict['remaining']) == 0:
            data_dict['remaining'] = [data_dict['disconnected'][1]]
            data_dict['disconnected'] = [data_dict['disconnected'][0]]
        
        if len(data_dict['disconnected']) == 0 and len(data_dict['remaining']) == 2:
            data_dict['disconnected'] = [data_dict['remaining'][0]]
            data_dict['remaining'] = [data_dict['remaining'][1]]


        # pretty print the data_dict
        # logging.info(f"Data dict: {json.dumps(data_dict, indent=4)}")


        messages = build_chat_messages_with_disconnected(instruction_prompt, data_dict, 
                                                        disconnected_preamble, remaining_preamble)
        return messages 
    
    elif prompt_type == "flattened":
        template = PROMPT_TEMPLATES["prompt_benchmarking_flattened_style"]
        instruction_prompt = template.render()
        messages = build_chat_messages_with_disconnected(instruction_prompt, data_dict, 
                                                        disconnected_preamble, remaining_preamble)
        messages = flatten_messages_to_prompt(messages)
        
        return messages

async def prepare_messages_file(conflicts, instances_dict, messages_file_path, prompt_type):
    '''
    For benchmarking of the disambiguation process.
    '''
    count = 0
    for key in conflicts:
        count += 1
        logging.info(f"Building messsages for conflict {count} - {key}")
        try:
            full_conflict = await build_full_conflict(conflicts[key], instances_dict)
            messages = build_prompt_benchmarking(full_conflict, prompt_type)
            logging.info(f"Number of messages: {len(messages)}")
            result = {
                key : messages
            }
            write_to_results_file(result, messages_file_path)
            logging.info(f"Messages for conflict {key} written to file.")
        
        except Exception as e:
            logging.error(f"Error processing conflict {key}: {e}")
            raise

# ---------------------------------------------------------------
# MAKE INFERENCES: with a given model for a given set of messages 
# -------------------------------------------------------------
def parse_messages_file(messages_file_path):
    messages = {}
    with open(messages_file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    key = next(iter(entry))
                    message = entry[key]
                    messages[key] = message
                    
                except Exception as e:
                    logging.warning(f"Could not parse line: {line[:100]}...\n{e}")

    return messages

if __name__ == "__main__":

    results_paths = {
        "model_1": "scripts/data/model_1_results.json",
        "model_2": "scripts/data/model_2_results.json",
        # Add more models as needed
    }

    # ---------- JUPYTER NOTEBOOK 1 - METRICS CALCULATION ----------------------
    models = []
    # Compare with human results and compute metrics
    human_results = "scripts/data/human_results.json"
    for model in models:
        logging.info(f"Comparing results for model {model}")
        # Load the results
        with open(results_paths[model], 'r') as f:
            results = [json.loads(line) for line in f]
        
        # Load the human results
        with open(human_results, 'r') as f:
            human_results = [json.loads(line) for line in f]

        # Compare the results with human results
        # Compute metrics such as accuracy, precision, recall, F1-score, etc.
        # Put in dataframe
        # Plot the results
        
        pass

    # 4. Save results to file

