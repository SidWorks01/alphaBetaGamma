#!/usr/bin/env python3
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

# Get absolute path to ../prompts (same as yours)
current_dir = os.path.dirname(__file__)
prompts_path = os.path.join(current_dir, "..", "prompts")
sys.path.append(os.path.abspath(prompts_path))

from module_1_prompt import SYSTEM_PROMPT

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"        # choose your model
INPUT_COL = "transcript"          # column with {"0": "Agent: ...", ...}
OUTPUT_COL = "annotated"          # column to store the new structured json
SAVE_EVERY = 200                  # checkpoint frequency (not used in this small example)
OUTPUT_PATH = "df_turned_annotated.parquet"  # kept for compatibility if you want to switch
NUM_WORKERS = 5                  # number of threads

input_file = 'data/input.csv'
output_file = 'data/output.csv'

# If you want to set a default max number of rows to process, change this variable.
# You can also pass --n on the command line to override.
PROCESS_N = None  # None means "process all eligible rows"


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
                response_format={"type": "json_object"},
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


def row_already_processed(row: list) -> bool:
    """
    Determine whether this input row is already annotated.
    We treat a row as processed if it has at least 4 columns and
    both the inferenced_metadata (index 2) and labels (index 3) are non-empty strings.
    """
    try:
        if len(row) >= 4:
            if str(row[2]).strip() and str(row[3]).strip():
                return True
    except Exception:
        pass
    return False


def process_row(row_idx: int, row: list, api_key: str, write_lock: threading.Lock, csv_writer, pbar):
    """
    Process a single CSV row; parse transcript from col0, call LLM, and write result row to csv_writer.
    Returns tuple (row_idx, True/False, error_message_or_None)
    """
    try:
        if len(row) < 1:
            return (row_idx, False, "empty row")

        # If row already looks annotated, skip (safety)
        if row_already_processed(row):
            return (row_idx, True, "already processed - skipped")

        col1 = row[0]
        col2 = row[1] if len(row) > 1 else ""

        # transcript is expected to be a JSON string in col1
        transcript_json = col1
        transcript = json.loads(transcript_json)

        # Create a client per thread (your approach)
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


def load_processed_transcripts_from_output(output_path: str) -> set:
    """
    If output CSV exists, read it and return a set of transcript JSON strings
    (the value in column 0) that are already present in the output file.
    """
    processed = set()
    if not os.path.exists(output_path):
        return processed

    try:
        with open(output_path, mode='r', newline='', encoding='utf-8') as f_out:
            reader = csv.reader(f_out)
            # skip header if present
            try:
                header = next(reader)
            except StopIteration:
                return processed
            # read rows: transcript assumed to be column 0
            for row in reader:
                if not row:
                    continue
                transcript_val = row[0]
                if transcript_val is not None:
                    processed.add(transcript_val)
    except Exception as e:
        print(f"[WARNING] Failed to read existing output file {output_path}: {e}. Continuing as if empty.")
    return processed


def main_iteration(max_rows: int = None):
    """
    Read input_file, skip rows that already have inferenced metadata and labels,
    skip rows that are already present in output.csv (if it exists),
    and process up to max_rows rows (if provided). Results are appended to output_file.
    """
    # Read entire input CSV (including header)
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    with open(input_file, mode='r', newline='', encoding='utf-8') as f_in:
        reader = list(csv.reader(f_in))

    if not reader:
        print("Input CSV is empty.")
        return

    # Assume first row is input header - data starts from row 1
    input_header = reader[0]
    data_rows = reader[1:]

    # Load already-processed transcripts from existing output file (if any)
    processed_transcripts = load_processed_transcripts_from_output(output_file)
    if processed_transcripts:
        print(f"Found {len(processed_transcripts)} transcripts already present in {output_file}. These will be skipped.")

    # Determine which rows need processing: skip rows that are already annotated in input OR already in output.csv
    rows_to_submit = []
    for idx, row in enumerate(data_rows, start=1):  # start=1 to match original file row index semantics
        # If input row itself claims to be already processed (has columns 2 & 3), skip
        if row_already_processed(row):
            continue
        # If the transcript (col0) already exists in output.csv, skip
        if len(row) >= 1 and row[0] in processed_transcripts:
            continue
        rows_to_submit.append((idx, row))

    if max_rows is not None:
        rows_to_submit = rows_to_submit[:max_rows]

    total_to_process = len(rows_to_submit)
    if total_to_process == 0:
        print("No rows to process (either already annotated in input or already present in output.csv).")
        return

    # Prepare output file for appending new rows.
    # If the file doesn't exist yet, we'll create it and write the header.
    write_lock = threading.Lock()
    need_write_header = not os.path.exists(output_file)
    out_file = open(output_file, mode='a', newline='', encoding='utf-8')  # append mode
    writer = csv.writer(out_file)

    if need_write_header:
        writer.writerow(["transcript", "metadata", "inferenced_metadata", "labels"])

    pbar = tqdm(total=total_to_process, desc="Annotating", unit="row")

    futures = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for row_idx, row in rows_to_submit:
            futures.append(executor.submit(process_row, row_idx, row, API_KEY, write_lock, writer, pbar))

        for fut in as_completed(futures):
            row_idx, success, err = fut.result()
            if not success:
                print(f"[ERROR] row {row_idx}: {err}")

    pbar.close()
    out_file.close()
    print(f"CSV processing completed! Appended {total_to_process} new rows to {output_file}.")

if __name__ == "__main__":
    # change max_rows as you like; using 10 here as an example
    max_rows = 19621
    main_iteration(max_rows=max_rows)
