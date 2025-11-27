import os
import json
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import sys
import os

# Get absolute path to ../prompts
current_dir = os.path.dirname(__file__)
prompts_path = os.path.join(current_dir, "..", "prompts")

sys.path.append(os.path.abspath(prompts_path))

from module_1_prompt import SYSTEM_PROMPT

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# API_KEY_PATH = "openai_key"
# with open(API_KEY_PATH, 'r') as key_file:
#     API_KEY = key_file.read().strip()

MODEL_NAME = "gpt-4o-mini"        # choose your model
INPUT_COL = "transcript"          # column with {"0": "Agent: ...", ...}
OUTPUT_COL = "annotated"          # column to store the new structured json
SAVE_EVERY = 200                  # checkpoint frequency (not used in this small example)
OUTPUT_PATH = "df_turned_annotated.parquet"  # kept for compatibility if you want to switch
NUM_WORKERS = 20                  # number of threads

input_file = 'data/input.csv'
output_file = 'data/output.csv'

def annotate_single_transcript(turn_dict: dict, client: OpenAI, model_name: str = MODEL_NAME) -> dict:
    """
    turn_dict: {"0": "Agent: ...", "1": "Customer: ...", ...}
    returns:   structured dict matching the "new transcript format"
    NOTE: this version expects a thread-local 'client' (OpenAI instance).
    """
    turns_sorted = {str(k): turn_dict[k] for k in sorted(turn_dict.keys(), key=lambda x: int(x))}
    conv_text = json.dumps(turns_sorted, ensure_ascii=False, indent=2)

    user_prompt = f"""
Here is the transcript as a JSON object (mapping turn IDs to utterances):

{conv_text}
Return ONLY valid JSON. Do not add any surrounding text, comments, or code fences.
Use double quotes for all strings and property names, and do not include trailing commas.

Follow ALL instructions in the system message and return ONLY the final JSON object in the required output format.
"""

    backoff = 1
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)  # should match the required schema
            return data

        except json.JSONDecodeError:
            print("JSON parse error, raw response:")
            print(raw)
            raise

        except Exception as e:
            print(f"[annotate_single_transcript] Error: {e}. Backing off {backoff}s and retrying...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


def process_row(row_idx: int, row: list, api_key: str, write_lock: threading.Lock, csv_writer, pbar):
    """
    Process a single CSV row; parse transcript from col0, call LLM, and write result row to csv_writer.
    """
    try:
        if len(row) < 1:
            return (row_idx, False, "empty row")

        col1 = row[0]
        col2 = row[1] if len(row) > 1 else ""

        transcript_json = col1
        transcript = json.loads(transcript_json)

        client = OpenAI(api_key=api_key)

        response = annotate_single_transcript(transcript, client)

        inferenced_metadata = response.get('inferred_metadata', {})
        labels = response.get('transcript', {})

        inferenced_metadata_json = json.dumps(inferenced_metadata, ensure_ascii=False, indent=4)
        labels_json = json.dumps(labels, ensure_ascii=False, indent=4)

        new_row = [col1, col2, inferenced_metadata_json, labels_json]

        with write_lock:
            csv_writer.writerow(new_row)

        return (row_idx, True, None)

    except Exception as e:
        return (row_idx, False, str(e))
    finally:
        pbar.update(1)


def main_iteration():
    with open(input_file, mode='r', newline='', encoding='utf-8') as f_in:
        reader = list(csv.reader(f_in))
    total_rows = len(reader)

    write_lock = threading.Lock()
    out_file = open(output_file, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(out_file)

    writer.writerow(['col1', 'col2', 'inferenced_metadata', 'labels'])

    pbar = tqdm(total=total_rows, desc="Annotating", unit="row")

    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for idx, row in enumerate(reader):
            futures.append(executor.submit(process_row, idx, row, API_KEY, write_lock, writer, pbar))

        for fut in as_completed(futures):
            row_idx, success, err = fut.result()
            if not success:
                print(f"[ERROR] row {row_idx}: {err}")

    pbar.close()
    out_file.close()
    print("CSV processing completed!")


def sample_testing():
    # Read from input CSV
    input_file = 'data/input.csv'
    output_file = 'data/output.csv'

    # Open input file for reading
    input_csv = open(input_file, mode='r', newline='')
    reader = csv.reader(input_csv)
    client = OpenAI(api_key=API_KEY)
    # Open output file for writing
    output_csv = open(output_file, mode='w', newline='')
    writer = csv.writer(output_csv)
    # Process each row
    row_test = next(reader)
    for row in reader:
        if len(row) >= 2:  # Check if row has at least 2 columns
            col1 = row[0]
            col2 = row[1]
            transcript_json = row[0]   # Get column 1
            transcript = json.loads(transcript_json)
            response = annotate_single_transcript(transcript, client)
            inferenced_metadata = response['inferred_metadata']
            labels = response['transcript']
            inferenced_metadata_json = json.dumps(inferenced_metadata, indent=4)
            labels_json = json.dumps(labels, indent=4)
        
            # Create new row with original data + results
            new_row = [col1, col2, inferenced_metadata_json, labels_json]
            
            # Write to output file
            writer.writerow(new_row)
            break

    # Close both files
    input_csv.close()
    output_csv.close()

    print("CSV processing completed!")


if __name__ == "__main__":
    sample_testing()