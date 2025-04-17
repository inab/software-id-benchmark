import json
import pandas as pd
from src.application.services.benchmarking import prepare_messages_file


def build_instances_keys_dict(data):
    """Create a mapping of instance IDs to their respective instance data."""
    instances_keys = {}
    for key, value in data.items():
        instances = value.get("instances", [])
        for instance in instances:
            instances_keys[instance["_id"]] = instance
    return instances_keys


async def prepare_messages(grouped_entries_file, evaluation_cases_csv, messages_file, prompt_type):
    """
    Prepare messages for benchmarking the disambiguation process.
    - prompt_type: "chat" or "flattened"
    """
    # Load grouped entries
    with open(grouped_entries_file, "r", encoding="utf-8") as f:
        grouped_entries = json.load(f)

    # Create instances dict from grouped entries
    instances_dict = build_instances_keys_dict(grouped_entries)


    # Load evaluation cases
    valuation_cases = pd.read_csv(evaluation_cases_csv)
    evaluation_disconnected_entries = {}
    for index, row in valuation_cases.iterrows():
        case_id = row['entry_id']
        entry_1_id = row['entry_1_id']
        entry_2_id = row['entry_2_id']
        evaluation_disconnected_entries[case_id] = {}
        entry_1 = {
            'id': entry_1_id,
            'name': instances_dict[entry_1_id]['data']['name'],
            'description': instances_dict[entry_1_id]['data']['description'],
            'repository': instances_dict[entry_1_id]['data']['repository'],
            'webpage': instances_dict[entry_1_id]['data']['webpage'],
            'license': instances_dict[entry_1_id]['data']['license'],
            'authors': instances_dict[entry_1_id]['data']['authors'],
            'publication': instances_dict[entry_1_id]['data']['publication'],
            'source': instances_dict[entry_1_id]['data']['source'],
        }

        entry_2 = {
            'id': entry_2_id,
            'name': instances_dict[entry_2_id]['data']['name'],
            'description': instances_dict[entry_2_id]['data']['description'],
            'repository': instances_dict[entry_2_id]['data']['repository'],
            'webpage': instances_dict[entry_2_id]['data']['webpage'],
            'license': instances_dict[entry_2_id]['data']['license'],
            'authors': instances_dict[entry_2_id]['data']['authors'],
            'publication': instances_dict[entry_2_id]['data']['publication'],
            'source': instances_dict[entry_2_id]['data']['source'],
        }

        evaluation_disconnected_entries[case_id]['disconnected'] = [entry_1]
        evaluation_disconnected_entries[case_id]['remaining'] = [entry_2]


    # Create messages for each inference
    await prepare_messages_file(evaluation_disconnected_entries, instances_dict, messages_file, prompt_type)
   
    return messages_file



if __name__ == "__main__":
    disconnected_entries_file = 'scripts/data/evaluation/disconnected_entries.json'
    messages_file = 'scripts/data/evaluation/messages.json'
    grouped_entries_file = 'data/grouped.json'
    evaluation_cases = 'scripts/data/evaluation/software_disambiguation_evaluation_conflicts.csv'

    # create instances dict from grouped entries 
    instances_dict = build_instances_keys_dict(grouped_entries_file)

    # create disconnected entries dict from disconnected entries
    with open(disconnected_entries_file, "r", encoding="utf-8") as f:
        disconnected_entries = json.load(f)

    # create disconnected entries dict from evaluation cases
    valuation_cases = pd.read_csv(evaluation_cases)
    evaluation_disconnected_entries = {}
    for index, row in valuation_cases.iterrows():
        case_id = row['entry_id']
        evaluation_disconnected_entries[case_id] = disconnected_entries[case_id]

    # Create messages for each inference
    prepare_messages_file(evaluation_disconnected_entries, instances_dict, messages_file)


