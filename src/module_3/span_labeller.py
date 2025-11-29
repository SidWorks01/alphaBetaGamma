import json
import uuid
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import math

load_dotenv()
k = 2

ALL_BUSINESS_LABELS = [
    "BRQ", "BRE", "BRS", "BIS", "BIR", "BRV", "BPU", "BTR", "BFR", "BNW",
    "GENQ", "GRES", "POLI", "HOLD", "META", "NOISE", "OFFT", "ERR",
    "CRIT", "NCRT"
]

ALL_CONV_MARKERS = [
    "FRU", "ANX", "DIS", "POS", "THK", "DEC", "HES", "SEC", "CLM", "INF",
    "QUE", "ACK", "AFF", "EMP", "PRO", "OFF", "REP", "ESL", "NEG", "SWI",
    "CH0", "CH1", "CH2", "CH3", "CH4"
]


def clean_json(s: str):
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    s = s.replace('""', '"')
    return s

def one_hot(labels, ref_list):
    vec = np.zeros(len(ref_list))
    if not labels:
        return vec
    if isinstance(labels, str):
        labels = [labels]
    for l in labels:
        if l in ref_list:
            vec[ref_list.index(l)] = 1
    return vec

def get_top_k(vec, ref_list, turns):
    idx = np.argsort(vec)[::-1]
    result = []
    for i in idx[:k]:
        if vec[i] > 0:
            for t in turns:
                if ref_list[i] in t.get('turn_business_label', []) or \
                   ref_list[i] in t.get('turn_conv_marker', []):
                    result.append(ref_list[i])
                    break
    result = list(dict.fromkeys(result))
    return result

def filter_entities(global_entities, active_indices):
    output = {}
    if not isinstance(global_entities, dict):
        return output
    for category, items in global_entities.items():
        cat = {}
        for name, idx_list in items.items():
            valid = [idx for idx in idx_list if idx in active_indices]
            if valid:
                cat[name] = valid
        if cat:
            output[category] = cat
    return output

def process_span_flow(turns, meta):
    global_entities = meta.get("entity", {})
    active_indices = [str(t.get("turn_idx")) for t in turns]
    print(active_indices)

    b_vecs, c_vecs = [], []

    for t in turns:
        causal = t.get("causal_score", 0)

        b = one_hot(t.get("turn_business_label", []), ALL_BUSINESS_LABELS) * causal
        c = one_hot(t.get("turn_conv_marker", []), ALL_CONV_MARKERS) * causal

        b_vecs.append(b)
        c_vecs.append(c)

    final_b = np.max(b_vecs, axis=0)
    final_c = np.max(c_vecs, axis=0)

    span_flow_meta = {
        "category_vector": final_b.tolist() + final_c.tolist(),
        "id": str(uuid.uuid4()),
        "turn_idx": active_indices,
        "label": {
            "business_label": get_top_k(final_b, ALL_BUSINESS_LABELS, turns),
            "conversational_markers": get_top_k(final_c, ALL_CONV_MARKERS, turns),
            "entity": filter_entities(global_entities, active_indices)
        }
    }
    return span_flow_meta

csv_path = os.getenv("DATASET_PATH")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

if "span_flow_metadata" not in df.columns:
    df["span_flow_metadata"] = None
for idx, row in df.iterrows():

    raw_analysis = row.get("analysis")
    raw_meta = row.get("inferenced_metadata")

    if raw_analysis is None or (isinstance(raw_analysis, float) and math.isnan(raw_analysis)):
        continue
    if raw_meta is None or (isinstance(raw_meta, float) and math.isnan(raw_meta)):
        continue

    if isinstance(raw_analysis, str):
        analysis_list = json.loads(raw_analysis)
    elif isinstance(raw_analysis, list):
        analysis_list = raw_analysis
    else:
        continue

    if isinstance(raw_meta, str):
        meta_clean = clean_json(raw_meta)
        meta_dict = json.loads(meta_clean)
    elif isinstance(raw_meta, dict):
        meta_dict = raw_meta
    else:
        continue

    if raw_analysis is None or meta_dict is None:
        print(f"Skipping row {idx}: Could not load analysis or metadata")
        continue

    span_analysis_list = analysis_list["spans"]
    flow_analysis_dict = analysis_list["flows"]

    all_span_metadata = []
    count_span = 0

    for span_block in span_analysis_list:
        span_meta = process_span_flow(span_block, meta_dict)
        all_span_metadata.append(span_meta)
        count_span += 1

    all_flow_metadata = []
    count_flow = 0

    for flow_key, flow_turns in flow_analysis_dict.items():
        flow_meta = process_span_flow(flow_turns, meta_dict)
        all_flow_metadata.append({flow_key: flow_meta})
        count_flow += 1

    df.at[idx, "span_flow_metadata"] = json.dumps({
        "spans": all_span_metadata,
        "flows": all_flow_metadata
    })

    print(f"Row {idx}: spans={count_span}, flows={count_flow} processed")


df.to_csv(csv_path, index=False)
print("\n✔ ALL ROWS & ALL SPANS/FLOWS PROCESSED SUCCESSFULLY")