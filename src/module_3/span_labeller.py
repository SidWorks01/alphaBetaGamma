import json
import uuid
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
k = 2

ALL_BUSINESS_LABELS = [
    "BRQ", "BRE", "BRS", "BIS", "BIR", "BRV", "BPU", "BTR", "BFR", "BNW",
    "GENQ", "GRES", "POLI", "HOLD", "META", "NOISE", "OFFT", "REP", "ERR"
]

ALL_CONV_MARKERS = [
    "FRU", "ANX", "DIS", "POS", "THK", "DEC", "HES", "SEC", "CLM", "INF",
    "QUE", "ACK", "AFF", "EMP", "PRO", "OFF", "REP", "ESL", "NEG", "SWI",
    "CH0", "CH1", "CH2", "CH3", "CH4"
]


def clean_metadata_json(s: str):
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
    return list(dict.fromkeys(result))

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

def process_span(turns, meta):
    global_entities = meta.get("entity", {})
    active_indices = [str(t.get("turn_idx")) for t in turns]

    b_vecs, c_vecs = [], []

    for t in turns: # for every turn
        causal = t.get("causal_score", 0)
        b_vecs.append(one_hot(t.get("turn_business_label", []), ALL_BUSINESS_LABELS) * causal) #final_business_vector
        c_vecs.append(one_hot(t.get("turn_conv_marker", []), ALL_CONV_MARKERS) * causal) #final_conv_vector

    final_b = np.max(b_vecs, axis=0)
    final_c = np.max(c_vecs, axis=0)

    span_meta = {
        "category_vector": final_b.tolist() + final_c.tolist(), #category_vector
        "span_id": str(uuid.uuid4()),
        "span_label": {
            "business_label": get_top_k(final_b, ALL_BUSINESS_LABELS, turns),
            "conversational_markers": get_top_k(final_c, ALL_CONV_MARKERS, turns),
            "entity": filter_entities(global_entities, active_indices)
        }
    }
    return span_meta


csv_path = os.getenv("DATASET_PATH")
df = pd.read_csv(csv_path)

target_index = df.index[-1]

analysis_list = json.loads(df.at[target_index, "Analysis"])

meta_raw = clean_metadata_json(df.at[target_index, "Inferred_Metadata"])
meta_list = json.loads(meta_raw)

all_span_metadata = []
for span_block in analysis_list:
    if isinstance(span_block, dict):
        span_block = [span_block]
    span_meta = process_span(span_block, meta_list)
    all_span_metadata.append(span_meta)

df.at[target_index, "span_metadata"] = json.dumps(all_span_metadata)
df.to_csv(csv_path, index=False)

print("\n Generated span metadata for ALL spans in analysis_list for row:", target_index)
print(json.dumps(all_span_metadata, indent=4))
