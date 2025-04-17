import requests
import json
import re
import os
import logging
from pathlib import Path
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from jinja2 import Template
from functools import lru_cache
from readability import Document
from bs4 import BeautifulSoup
import tiktoken

from src.application.services.enrich_links import enrich_link

# -------------------------------
# Configuration
# -------------------------------

# OpenRouter - https://openrouter.ai
OR_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OR_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.3-70b-instruct:free"
REQUESTS_PER_MINUTE = 20
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE
MAX_TOTAL_TOKENS = 130000  


# HuggingFace - https://huggingface.co
HF_API_URL = "https://api-inference.huggingface.co/models" 
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")




logging.basicConfig(level=logging.INFO)

# -------------------------------
# Template Loader
# -------------------------------

def load_templates_from_folder(folder_path: str) -> dict:
    logging.info(f"Loading templates from folder: {folder_path}")
    templates = {}
    for file in Path(folder_path).glob("*.jinja2"):
        key = file.stem
        logging.info(f"Loading template: {key}")
        with open(file, encoding="utf-8") as f:
            templates[key] = Template(f.read())
    return templates

PROMPT_TEMPLATES = load_templates_from_folder("src/application/services/integration/prompts")

# -------------------------------
# Tokenization Utilities
# -------------------------------

@lru_cache
def get_tokenizer(model="gpt-4"):
    return tiktoken.encoding_for_model(model)

def count_tokens(text, model="gpt-4") -> int:
    return len(get_tokenizer(model).encode(text))

def estimate_total_tokens(messages, model="gpt-4"):
    enc = get_tokenizer(model)
    return sum(len(enc.encode(msg["content"])) for msg in messages)


# -------------------------------
# Chunking big text
# -------------------------------
def chunk_text(text: str, max_tokens: int = 8000, model: str = "gpt-4"):
    enc = get_tokenizer(model)
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        token_len = len(enc.encode(word + " "))
        if current_token_count + token_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_token_count = token_len
        else:
            current_chunk.append(word)
            current_token_count += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -------------------------------
# Prompt + Chat Message Builder
# -------------------------------

def build_chat_messages_with_disconnected(
    instruction_prompt: str,
    conflict_data: dict,
    disconnected_preamble= "**Disconnected tools** to be analyzed",
    remaining_preamble= "Tools known to be the **same software**",
    max_tokens_per_chunk=8000,
    model="gpt-4"
):
    messages = [{"role": "user", "content": instruction_prompt}]
    enc = get_tokenizer(model)

    def chunk_entries(entries: List[dict]):
        chunks = []
        current_chunk = []
        current_token_count = 0

        for entry in entries:
            entry_json = json.dumps(entry, ensure_ascii=False)
            entry_tokens = len(enc.encode(entry_json))

            if current_token_count + entry_tokens > max_tokens_per_chunk:
                chunks.append(current_chunk)
                current_chunk = [entry]
                current_token_count = entry_tokens
            else:
                current_chunk.append(entry)
                current_token_count += entry_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_dict(d: dict):
        for url, content in d.items():
            for label, body in content.items():
                text = f"Content from {url}:\n```\n{label}:\n{body}```"
                yield {"role": "user", "content": text}

    # Add entries: known and disconnected tools
    if conflict_data.get("remaining"):
        for i, chunk in enumerate(chunk_entries(conflict_data["remaining"])):
            messages.append({
                "role": "user",
                "content": f"{remaining_preamble} - part {i+1}:\n```json\n{json.dumps(chunk, indent=2, ensure_ascii=False)}\n```"
            })

    if conflict_data.get("disconnected"):
        for j, chunk in enumerate(chunk_entries(conflict_data["disconnected"])):
            messages.append({
                "role": "user",
                "content": f"{disconnected_preamble} - part {j+1}:\n```json\n{json.dumps(chunk, indent=2, ensure_ascii=False)}\n```"
            })

    # Add enriched webpage and repository contents
    if "webpage_contents" in conflict_data:
        for chunk in chunk_dict(conflict_data["webpage_contents"]):
            messages.append(chunk)

    # Final instruction
    messages.append({
        "role": "user",
        "content": "All parts have been sent. Please now analyze the entries and provide the output as specified. \n\nIMPORTANT: Return ONLY a valid Python dictionary with the following keys: 'verdict', 'explanation', 'confidence', and 'features'. Do NOT explanation, or extra commentary. This is a strict output constraint."
    })

    # Token budget check
    total_tokens = estimate_total_tokens(messages, model=model)
    logging.info(f"Total tokens: {total_tokens}")
    if total_tokens > MAX_TOTAL_TOKENS:
        raise ValueError(f"Prompt too long: {total_tokens} tokens. Limit is {MAX_TOTAL_TOKENS}.")

    return messages


# -------------------------------
# Prompt Selection
# -------------------------------

def build_prompt(disconnected, remaining):
    n_disconnected = len(disconnected)
    n_remaining = len(remaining)

    if n_disconnected == 1 and n_remaining == 1:
        logging.info("Using template: disconnected_entries")
        template = PROMPT_TEMPLATES["disconnected_entries"]
    elif n_disconnected > 1 and n_remaining == 0:
        logging.info("Using template: disconnected_entries")
        template = PROMPT_TEMPLATES["disconnected_entries"]
    elif n_disconnected == 1 and n_remaining > 1:
        logging.info("Using template: one_disconnected_several_remaining")
        template = PROMPT_TEMPLATES["one_disconnected_several_remaining"]
    elif n_disconnected > 1 and n_remaining > 1:
        logging.info("Using template: several_disconnected_several_remaining")
        template = PROMPT_TEMPLATES["several_disconnected_several_remaining"]
    else:
        raise ValueError(f"Unsupported combination: {n_disconnected} disconnected, {n_remaining} remaining")

    instruction_prompt = template.render()

    data_dict = {
        "disconnected": disconnected,
        "remaining": remaining
    }

    return build_chat_messages_with_disconnected(instruction_prompt, data_dict)

# -------------------------------
# API Request
# -------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_openrouter(messages, model):
    headers = {
        "Authorization": f"Bearer {OR_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2
    }

    logging.info(f"Sending request to OpenRouter API: {OR_API_URL} with key {OR_API_KEY[:4]}...")
    
    response = requests.post(OR_API_URL, json=payload, headers=headers )
    
    if response.status_code == 200:
        try:
            # main answer 
            content = response.json()["choices"][0]["message"]["content"].strip()
            # metadata
            meta = response.json().get("usage", {})
            meta['provider'] = response.json().get("provider", "")
            if content:
                return content, meta
        except:
            logging.warning(response.json())
    
    logging.warning(f"API response was empty: {response.status_code} - {response.text}")
    return '', {}




#@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_huggingface_new(messages, model, provider):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
     
    URL = f"https://router.huggingface.co/{provider}/v1/chat/completions"
    logging.info(f"Sending request to Hugging Face Inference API: {URL} with key {HF_API_KEY[:4]}...")
    
    response = requests.post(URL, headers=headers, json=payload)
    logging.info(f"API response: {response}")
    if response.status_code == 200:
        try:
            # main answer 
            content = response.json()["choices"][0]["message"]["content"].strip()
            # metadata
            meta = response.json().get("usage", {})
            meta['provider'] = provider
            if content:
                return content, meta
           
        except Exception as e:
            logging.warning(f"Parsing error: {e} | Response: {response.json()}")
        
    logging.warning("API response was empty after 3 attempts.")
    return '', {}

def query_huggingface(messages, model):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = { 
        "inputs": messages,
        "parameters": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_new_tokens": 512,
            "return_full_text": False
        }
    }
     
    URL = f"{HF_API_URL}/{model}"
    logging.info(f"Sending request to Hugging Face Inference API: {URL} with key {HF_API_KEY[:4]}...")
    
    response = requests.post(URL, headers=headers, json=payload)
    logging.info(f"API response: {response.json()}")
    if response.status_code == 200:
        try:
            print(f"Whole response: {response.json()}")
            output_text = response.json()[0]["generated_text"].strip()
            # TODO: extract metadata
            return output_text, {}  # You could add more metadata if needed
        except Exception as e:
            logging.warning(f"Parsing error: {e} | Response: {response.json()}")
        
    logging.warning("API response was empty after 3 attempts.")
    return None


# Process flagged cases
def make_inference(messages, model, provider):

    # Query LLM to check if they are the same
    if provider == "openrouter":
        query_func = query_openrouter
    elif provider == "huggingface":
        query_func = query_huggingface

    try: 
        #result, meta = query_openrouter(conflict, model)
        result, meta = query_func(messages, model=model)
    
    except Exception as e:
        logging.error(f"Error resolving conflict: {e}")
        return None
    
    return result, meta

# -------------------------------
# Conflict Handling
# -------------------------------

async def build_full_conflict(conflict, instances_dict, max_tokens=8000, model="gpt-4"):
    """
    Returns a conflict dictionary where:
    - 'disconnected' and 'remaining' contain minimal metadata (with URLs)
    - 'webpage_contents' and 'repository_contents' contain deduplicated content
    """
    from collections import defaultdict

    async def enrich_and_collect_content(url_list):
        contents = {}
        seen = set()

        for url in url_list:
            if url in seen or not url:
                continue
            seen.add(url)

            enriched = await enrich_link(url)

            contents[url] = {}

            if enriched and enriched.get("content"):
                chunks = chunk_text(enriched["content"], max_tokens=max_tokens, model=model)
                contents[url]["Content"]  = chunks

            if enriched and enriched.get("readme_content"):
                contents[url]["README content"] = chunk_text(enriched.get("readme_content"), max_tokens=max_tokens, model=model)
            
            if enriched and enriched.get("repo_metadata"):
                contents[url]['Repository metadata'] = enriched["repo_metadata"]
            
            if enriched and enriched.get("project_metadata"):
                contents[url]["Project metadata"] = enriched["project_metadata"]

        return dict(contents)


    new_conflict = {
        "disconnected": [],
        "remaining": [],
        "webpage_contents": {},
    }

    all_webpages = set()
    all_repos = set()

    def strip_content_fields(entry):
        entry = entry.copy()
        webpages = entry.get("webpage", [])
        repos = entry.get("repository", [])
        all_webpages.update(webpages)
        for repo in repos:  
            if repo.get('kind') == "github" and "github.com" in repo.get('url', ''):
                all_repos.add(repo.get('url', ''))
            elif repo.get('kind') == "bitbucket" and "bitbucket.com" in repo.get('url', ''):
                all_repos.add(repo.get('url', ''))
            elif repo.get('kind') == "gitlab" and "gitlab.com" in repo.get('url', ''):
                all_repos.add(repo.get('url', ''))
            else:
                all_webpages.add(repo.get('url', ''))
            
        return entry

    for entry in conflict["disconnected"]:
        new_conflict["disconnected"].append(strip_content_fields(entry))

    for entry in conflict["remaining"]:
        new_conflict["remaining"].append(strip_content_fields(entry))

    # Enrich and chunk all unique URLs
    repos_and_webpages = all_webpages.union(all_repos)
    all_links = set(repos_and_webpages)
    new_conflict["webpage_contents"] = await enrich_and_collect_content(list(all_links))

    return new_conflict


def parse_result(text):
    """
    Extracts and parses a JSON object from either a Markdown-style code block or raw inline JSON.

    Args:
        text (str): Input text containing the dictionary.

    Returns:
        dict: Parsed JSON object as a Python dictionary.

    Raises:
        ValueError: If no valid JSON is found or if JSON parsing fails.
    """

    # Try to extract from code block first
    match = re.search(r"```(?:json|python)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: try to find a top-level JSON object in plain text
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            logging.warning("No JSON object found in input.")
            return {}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON: {e}")
        return {}

def create_issue(issue):
    with open('data/issues.json', 'a') as f:
        f.write(json.dumps(issue, indent=4))


def log_error(conflict):
    with open('data/error_conflicts.json', 'a') as f:
        f.write(json.dumps(conflict, indent=4))


def log_result(result):
    with open('data/results.json', 'a') as f:
        f.write(json.dumps(result, indent=4))
    logging.info("Result logged")


def write_to_results_file(result, results_file):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, "a") as f:
            json.dump(result, f)
            f.write("\n")
    except Exception as e:
        logging.error(f"Error writing to results file: {e}")

def load_solved_conflict_keys(jsonl_path):
    solved_keys = set()
    if not os.path.exists(jsonl_path):
        return solved_keys
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    key = next(iter(entry))
                    solved_keys.add(key)
                except Exception as e:
                    logging.warning(f"Could not parse line: {line[:100]}...\n{e}")
    return solved_keys


def disambiguate_disconnected_entries(disconnected_entries, instances_dict, grouped_entries, results_file):
    '''
    Function for disambiguation in production. Scans grouped entries for disconnected entries and solve conflicts along the way.
    The result is a grouped dictionary with the same structure as the input, but with disambiguated entries.
    Also:
        - logs the results and creates issues for unclear cases.
        - loads the solved conflicts from the results file to avoid reprocessing them.
        - handles the case where the results file does not exist yet.
    '''

    disambiguated_grouped = {}
    results = {}
    count = 0
    solved_conflicts_keys = load_solved_conflict_keys(results_file)

    for key in grouped_entries:
        if key in disconnected_entries:
            if key not in solved_conflicts_keys:
                count += 1
                logging.info(f"Processing conflict {count} - {key}")
                try:
                    """ ------------------- PREPARE MESSAGES --------------------""" 

                    full_conflict = build_full_conflict(disconnected_entries[key], instances_dict)
                    messages = build_prompt(full_conflict["disconnected"], full_conflict["remaining"])
                    logging.info(f"Number of messages: {len(messages)}")

                    """ ------------------- DISAMBIGUATE --------------------"""
                    logging.info(f"Sending messages to OpenRouter for conflict {key}")
                    result = make_inference(messages)
                    parsed = parse_result(result)
                    if parsed:
                        logging.info(f"Result for conflict {key}: {parsed}")
                        results[key] = parsed
                        solved_conflicts_keys.add(key)
                        write_to_results_file({key: parsed}, results_file)

                        """ ------------------- ADD DISAMBIGUATED INTO GROUPED ENTRIES --------------------"""
                        if parsed["verdict"] != "Unclear":
                            disambiguated_grouped[key] = {
                                'instances': [[instances_dict[id] for id in group] for group in parsed["groups"]]
                            }
                        else:
                            create_issue(parsed['github_issue'])
                except Exception as e:
                    raise e
                    #logging.error(f"Error processing conflict {key}: {e}")
        else:
            """ ------------------- ADD INTO GROUPED ENTRIES --------------------"""
            disambiguated_grouped[key] = {'instances': grouped_entries[key]['instances']}

    return disambiguated_grouped, results


if __name__ == "__main__":
    disconnected_entries_file = 'data/disconnected_entries.json'
    instances_dict_file = 'data/instances_dict.json'
    grouped_entries_file = 'data/grouped.json'
    results_file = 'data/results.json'

