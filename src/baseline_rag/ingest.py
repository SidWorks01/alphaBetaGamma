# ingest.py
"""
Robust ingestion script for Chroma (PersistentClient) + OpenAI (new API).
Place dataset.json in project root under data/dataset.json or set DATASET_PATH env var.
Run: python src/baseline_rag/ingest.py
"""

import os
import json
import uuid
import time
import math
import random
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# OpenAI new client
from openai import OpenAI

# Chroma new client
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# -------------------- Config --------------------
# Default locations (adjust if needed)
PROJECT_ROOT = os.getcwd()                         # typically project root when using uv
DATASET_PATH = os.getenv("DATASET_PATH", "data/dataset.json")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
PERSIST_EVERY_N_BATCHES = int(os.getenv("PERSIST_EVERY_N_BATCHES", "1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
INITIAL_BACKOFF = float(os.getenv("INITIAL_BACKOFF", "1.0"))

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "transcripts")

# -------------------- Clients -------------------
# Use PersistentClient for on-disk storage (new Chroma API)
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Get or create collection
try:
    collection = client.get_or_create_collection(COLLECTION_NAME)
except Exception:
    # fallback for some chroma builds
    collection = client.create_collection(COLLECTION_NAME)

# -------------------- Helpers --------------------
def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Character-based chunker with overlap."""
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    step = max_chars - overlap
    if step <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_MAX_CHARS")
    while i < n:
        chunk = text[i:i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks

def safe_embed_texts(texts: List[str], model: str = EMBED_MODEL, max_retries: int = MAX_RETRIES) -> List[List[float]]:
    """
    Local embedding using Hugging Face BGE (BAAI) models.
    - Caches tokenizer & model on first call to avoid reloading.
    - Performs tokenization (truncation to 512 tokens), mean-pooling with attention mask,
      and L2-normalization to produce unit vectors (useful for cosine search with inner-product).
    - Keeps the same retry/backoff behavior as the original function (for transient local errors).
    Returns: List of float lists (embeddings) in the same order as `texts`.
    """
    import time
    import random
    from transformers import AutoTokenizer, AutoModel
    import torch

    # Static cache on the function object so we don't re-load every call
    if not hasattr(safe_embed_texts, "_hf_cached"):
        safe_embed_texts._hf_cached = {}

    cache = safe_embed_texts._hf_cached

    # Load model & tokenizer if needed (cache by model name)
    if model not in cache:
        print(f"[embed-local] Loading HF model/tokenizer for '{model}' (this may take a while)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
            hf_model = AutoModel.from_pretrained(model)
            # move model to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hf_model.to(device)
            hf_model.eval()
            cache[model] = {"tokenizer": tokenizer, "model": hf_model, "device": device}
        except Exception as exc:
            raise RuntimeError(f"Failed to load model '{model}': {exc}") from exc

    tokenizer = cache[model]["tokenizer"]
    hf_model = cache[model]["model"]
    device = cache[model]["device"]

    attempt = 0
    backoff = INITIAL_BACKOFF

    # internal helper: mean pooling + normalization
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_embeddings: (B, T, D), attention_mask: (B, T)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = summed / counts
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    while True:
        try:
            all_embs = []
            # We'll process in micro-batches to avoid OOM if the incoming `texts` list is large.
            # The ingest loop already batches by BATCH_SIZE, but be defensive here.
            micro_batch = 32
            for i in range(0, len(texts), micro_batch):
                batch = texts[i : i + micro_batch]
                toks = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                input_ids = toks["input_ids"].to(device)
                attention_mask = toks["attention_mask"].to(device)
                with torch.no_grad():
                    out = hf_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    # out.last_hidden_state: (B, T, D)
                    pooled = mean_pooling(out.last_hidden_state, attention_mask)  # (B, D)
                    emb_cpu = pooled.cpu().numpy()
                for row in emb_cpu:
                    all_embs.append(row.tolist())
            if len(all_embs) != len(texts):
                raise RuntimeError(f"Embedding length mismatch: got {len(all_embs)} embeddings for {len(texts)} texts")
            return all_embs

        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Local embeddings failed after {max_retries} retries: {exc}") from exc
            jitter = random.random() * 0.5
            wait = backoff + jitter
            print(f"[embed-local] attempt {attempt} failed: {exc}. Retrying in {wait:.1f}s ...")
            time.sleep(wait)
            backoff *= 2

def load_dataset(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put dataset.json at this path or set DATASET_PATH env var.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("dataset.json must be a JSON array of transcript records")
    return data

def transcript_to_chunk_objs(record: Dict) -> List[Dict]:
    conv = record.get("conversation", [])
    joined = "\n".join([f"{t['speaker']}: {t['text']}" for t in conv])
    chunks = chunk_text(joined)
    objs = []
    for idx, ch in enumerate(chunks):
        obj = {
            "id": str(uuid.uuid4()),
            "text": ch,
            "metadata": {
                "transcript_id": record.get("transcript_id"),
                "time_of_interaction": record.get("time_of_interaction"),
                "domain": record.get("domain"),
                "intent": record.get("intent"),
                "reason_for_call": record.get("reason_for_call"),
                "chunk_index": idx
            }
        }
        objs.append(obj)
    return objs

# -------------------- Ingestion --------------------
def ingest_all(dataset_path: str):
    print("Loading dataset:", dataset_path)
    data = load_dataset(dataset_path)
    print(f"Loaded {len(data)} transcripts")

    all_chunk_objs: List[Dict] = []
    for rec_idx, rec in enumerate(data):
        chunk_objs = transcript_to_chunk_objs(rec)
        all_chunk_objs.extend(chunk_objs)
        if (rec_idx + 1) % 50 == 0:
            print(f"Prepared chunks from {rec_idx+1}/{len(data)} transcripts...")

    total_chunks = len(all_chunk_objs)
    print(f"Total chunks to embed & upsert: {total_chunks}")

    for start in range(0, total_chunks, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_chunks)
        batch_objs = all_chunk_objs[start:end]
        texts = [o["text"] for o in batch_objs]
        ids = [o["id"] for o in batch_objs]
        metadatas = [o["metadata"] for o in batch_objs]

        print(f"[batch {start}-{end}] Creating embeddings for {len(texts)} texts ...")
        embeddings = safe_embed_texts(texts, model=EMBED_MODEL)

        if not (len(embeddings) == len(texts) == len(ids) == len(metadatas)):
            raise RuntimeError("Length mismatch between embeddings/texts/ids/metadatas")

        print(f"[batch {start}-{end}] Upserting to Chroma collection '{COLLECTION_NAME}' ...")
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

# -------------------- Main --------------------
if __name__ == "__main__":
    ingest_all(DATASET_PATH)
