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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env file or environment")

# Default locations (adjust if needed)
PROJECT_ROOT = os.getcwd()                         # typically project root when using uv
DATASET_PATH = os.getenv("DATASET_PATH", "dataset.json")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
PERSIST_EVERY_N_BATCHES = int(os.getenv("PERSIST_EVERY_N_BATCHES", "1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
INITIAL_BACKOFF = float(os.getenv("INITIAL_BACKOFF", "1.0"))

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "transcripts")

# -------------------- Clients --------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
    """Call OpenAI embeddings with retries and exponential backoff. Returns list of embeddings in same order."""
    attempt = 0
    backoff = INITIAL_BACKOFF
    while True:
        try:
            resp = openai_client.embeddings.create(model=model, input=texts)
            embeddings = [item.embedding for item in resp.data]
            return embeddings
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Embeddings failed after {max_retries} retries: {exc}") from exc
            jitter = random.random() * 0.5
            wait = backoff + jitter
            print(f"[embed] attempt {attempt} failed: {exc}. Retrying in {wait:.1f}s ...")
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
