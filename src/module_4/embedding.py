import csv
import json
import math
from typing import List, Dict
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# ---------------- CONFIG ----------------
CSV_PATH = "/content/fianl.csv"   # your CSV
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "bge_chunks"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64                   # adjust down if you get OOM
NORMALIZE = True                  # keep embeddings normalized for cosine-like search
# ----------------------------------------

# --- load model on GPU (if available) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading model on", device)
model = SentenceTransformer(EMBED_MODEL, device=device)

# --- read CSV into list of dicts ---
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    transcripts_list = list(reader)
print("Rows loaded:", len(transcripts_list))

# --- build chunks safely from analysis field --- 
# This incorporates both 'spans' and 'flows' from your analysis JSON
chunks_texts: List[str] = []
chunks_meta: List[Dict] = []

for row_idx, transcript in enumerate(transcripts_list):
    analysis_str = transcript.get("analysis")
    if not analysis_str:
        # skip rows with empty/None analysis
        continue
    try:
        analysis = json.loads(analysis_str)
    except Exception as e:
        # skip malformed JSON but print a warning
        print(f"Skipping row {row_idx}: invalid JSON in analysis ({e})")
        continue

    # --- process spans (list-of-lists of turns) ---
    spans = analysis.get("span", [])
    for span_idx, span in enumerate(spans):
        # ensure span is iterable
        if not isinstance(span, (list, tuple)):
            span = [span]
        # build chunk text by adding a period after each turn and joining
        processed = " ".join([
            (turn.get("turn") if isinstance(turn, dict) else str(turn)).strip() + "."
            for turn in span
            if turn is not None
        ]).strip()
        if processed == "":
            continue
        chunks_texts.append(processed)
        meta = {
            "source_row": row_idx,
            "type": "span",
            "span_index": span_idx
        }
        # include optional identifying fields from CSV row
        if transcript.get("id"):
            meta["row_id"] = transcript.get("id")
        chunks_meta.append(meta)

    # --- process flows (dict of class -> list of turns) ---
    # flows may be a dict mapping class names to lists of turns
    flows = analysis.get("flows", {})
    if isinstance(flows, dict):
        for flow_idx, (cls, turns) in enumerate(flows.items()):
            # turns expected to be list of dicts with key 'turn_text' per original snippet
            if not isinstance(turns, (list, tuple)):
                # coerce single value to list
                turns = [turns]
            flow_text = " ".join([
                (t.get("turn_text") if isinstance(t, dict) else str(t)).strip() + "."
                for t in turns
                if t is not None
            ]).strip()
            if flow_text == "":
                continue
            chunks_texts.append(flow_text)
            meta = {
                "source_row": row_idx,
                "type": "flow",
                "flow_class": cls,
                "flow_index": flow_idx
            }
            if transcript.get("id"):
                meta["row_id"] = transcript.get("id")
            chunks_meta.append(meta)
    else:
        # if flows is present but not a dict (e.g., list), try to handle generically
        if isinstance(flows, (list, tuple)):
            for flow_idx, turns in enumerate(flows):
                if not isinstance(turns, (list, tuple)):
                    turns = [turns]
                flow_text = " ".join([
                    (t.get("turn_text") if isinstance(t, dict) else str(t)).strip() + "."
                    for t in turns
                    if t is not None
                ]).strip()
                if flow_text == "":
                    continue
                chunks_texts.append(flow_text)
                meta = {
                    "source_row": row_idx,
                    "type": "flow",
                    "flow_class": None,
                    "flow_index": flow_idx
                }
                if transcript.get("id"):
                    meta["row_id"] = transcript.get("id")
                chunks_meta.append(meta)

print("Built chunks (including spans & flows):", len(chunks_texts))

if len(chunks_texts) == 0:
    raise SystemExit("No chunks to embed — check your CSV and the 'analysis' field.")

# --- Prepare Chroma client and collection (persistent) ---
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"model": EMBED_MODEL}
)
print("Chroma collection ready:", COLLECTION_NAME)

# --- Batch embed + add to Chroma ---
def add_batches_to_chroma(texts: List[str], metadatas: List[Dict], batch_size: int = BATCH_SIZE):
    n = len(texts)
    # we'll create stable ids using row/indices if available in metadata to avoid collisions
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        batch_texts = texts[i:j]
        batch_metas = metadatas[i:j]

        # construct ids: prefer row_id/span/flow indices when available
        batch_ids = []
        for k, m in enumerate(batch_metas):
            base_idx = i + k
            if m.get("type") == "span":
                rid = m.get("row_id", f"row{m['source_row']}")
                sid = m.get("span_index")
                batch_ids.append(f"{rid}_span{sid}")
            elif m.get("type") == "flow":
                rid = m.get("row_id", f"row{m['source_row']}")
                cls = m.get("flow_class") or "noflowclass"
                fidx = m.get("flow_index")
                batch_ids.append(f"{rid}_flow{fidx}_{cls}")
            else:
                batch_ids.append(f"chunk_{i + k}")

        # embed with SentenceTransformer (model already on device)
        emb = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=False)
        emb = np.asarray(emb, dtype=np.float32)

        # normalize if desired (so IndexFlatIP / inner product approximates cosine)
        if NORMALIZE and emb.shape[0] > 0:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms

        # add to chroma (embeddings must be python lists)
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=emb.tolist()
        )
        print(f"Added {len(batch_texts)} vectors (chunks {i}..{j-1}) to Chroma")

# run the batch upload
add_batches_to_chroma(chunks_texts, chunks_meta, batch_size=BATCH_SIZE)

# final: print total count of vectors (approx)
all_ids = collection.get(include=["ids"])["ids"]
print("All done. Total vectors in collection (approx):", len(all_ids))

