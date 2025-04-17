import pandas as pd
import json
import os


def write_to_results_file(result, results_file):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, "a") as f:
            json.dump(result, f)
            f.write("\n")
    except Exception as e:
        print(f"Error writing to file {results_file}: {e}")


def transform_human_results(evaluation_cases_csv, results_file):
    valuation_cases = pd.read_csv(evaluation_cases_csv)
    
    for index, row in valuation_cases.iterrows():
        case_id = row['entry_id']
        human_decision = row['human_decision']
        human_confidende= row['human_confidence']
        human_rationale = row['human_rationale']

        line = {
            case_id: {
                "verdict": human_decision,
                "confidence": human_confidende,
                "explanation": human_rationale
            }
        }

        # Write to the results file
        write_to_results_file(line, results_file)
        print(f"Transformed human results for case {case_id} written to file.")


if __name__ == "__main__":
    # Define paths
    evaluation_cases_csv = "scripts/evaluation/software_disambiguation_human.csv"
    results_file = "scripts/evaluation/human_results.jsonl"

    # Transform human results
    transform_human_results(evaluation_cases_csv, results_file)
    print(f"Transformed human results saved to '{results_file}'.")
        