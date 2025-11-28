import os
import math
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI new client
from openai import OpenAI

# Chroma new client
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE


CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "transcripts")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")   # choose your chat model
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))  # approximate truncation

# --------- Clients ----------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(), tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.get_or_create_collection(COLLECTION_NAME)
    
def embed_query(query: str) -> List[float]:
    """
    Local HF embedding for a single query string. Caches tokenizer+model per model name.
    Returns: list[float] (L2-normalized embedding)
    """
    # lazy imports to avoid top-level cost if not used
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    import time
    import random

    # Use the global EMBED_MODEL (unchanged)
    model_name = EMBED_MODEL

    # Cache on the function object so we don't reload repeatedly
    if not hasattr(embed_query, "_hf_cache"):
        embed_query._hf_cache = {}

    cache = embed_query._hf_cache

    if model_name not in cache:
        try:
            print(f"[embed-query] Loading HF model/tokenizer for '{model_name}' (this may take a while)...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            hf_model = AutoModel.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hf_model.to(device)
            hf_model.eval()
            cache[model_name] = {"tokenizer": tokenizer, "model": hf_model, "device": device}
        except Exception as exc:
            raise RuntimeError(f"Failed to load HF model '{model_name}': {exc}") from exc

    tokenizer = cache[model_name]["tokenizer"]
    hf_model = cache[model_name]["model"]
    device = cache[model_name]["device"]

    # tokenization, forward, mean-pooling, normalize
    toks = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    with torch.no_grad():
        out = hf_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        token_embeddings = out.last_hidden_state  # (1, T, D)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = summed / counts  # (1, D)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)  # unit vector

    emb = normalized.cpu().numpy()[0].astype(float)  # numpy 1D array
    return emb.tolist()

def retrieve_chunks(query_emb: List[float], top_k: int = TOP_K, where: dict = None) -> List[Dict]:  
    if where and len(where) > 0:
        # valid filter
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k * 3,
            where=where
        )
    else:
        # no filter
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k * 3
        )

    docs = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        docs.append({
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i]
        })

    return docs

def dedupe_chunks(chunks: List[Dict], top_k: int) -> List[Dict]:
    """Remove duplicates by (transcript_id, chunk_index) preserving order and keep up to top_k."""
    seen = set()
    out = []
    for c in chunks:
        md = c.get("metadata", {})
        key = (md.get("transcript_id"), md.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= top_k:
            break
    return out

def build_prompt(chunks: List[Dict], user_question: str) -> Dict[str, str]:
    parts = []
    citations = []
    for c in chunks:
        md = c.get("metadata", {})
        tid = md.get("transcript_id", "unknown")
        idx = md.get("chunk_index", "?")
        parts.append(f"[{tid}|{idx}] {c.get('text')}")
        citations.append(f"[{tid}|{idx}]")
    context = "\n\n---\n\n".join(parts)
    # simple truncation if too long: keep the last MAX_CONTEXT_CHARS chars
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
    system = (
        "You are an assistant that answers questions only using the provided context. "
        "Do NOT hallucinate. If the answer is not in the context, say 'I don't know'. "
        "At the end, list the snippet citations you used as [transcript_id|chunk_index]."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{user_question}\n\nInstructions: Answer concisely and include citations."
    return {"system": system, "user": user, "citations": citations}

def call_chat(system: str, user: str, model: str = LLM_MODEL, max_tokens: int = 300) -> str:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

def answer_query(question: str, top_k: int = TOP_K, metadata_filter: dict = None) -> Dict:
    q_emb = embed_query(question)
    retrieved = retrieve_chunks(q_emb, top_k=top_k, where=metadata_filter)
    if not retrieved:
        return {"answer": "I don't know (no relevant snippets found).", "retrieved": []}
    deduped = dedupe_chunks(retrieved, top_k=top_k)
    prompt = build_prompt(deduped, question)
    answer = call_chat(prompt["system"], prompt["user"])
    # return answer and which citations were used (we can show citations list)
    used_citations = [f"{c['metadata'].get('transcript_id')}|{c['metadata'].get('chunk_index')}" for c in deduped]
    return {"answer": answer, "citations": used_citations, "retrieved": deduped}

# --------- Simple REPL for interactive queries ----------
if __name__ == "__main__":
    print("RAG query REPL — type a question, or 'quit' to exit.")
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        # optional: ask for domain filter
        dom = input("Optional domain filter (press enter to skip): ").strip()
        mf = {"domain": dom} if dom else None
        res = answer_query(q, top_k=TOP_K, metadata_filter=mf)
        print("\n=== Answer ===")
        print(res["answer"])
        print("\n=== Citations ===")
        for c in res["citations"]:
            print(f"[{c}]")
        print("\n=== Retrieved snippets (titles) ===")
        for idx, r in enumerate(res["retrieved"]):
            md = r.get("metadata", {})
            print(f"{idx+1}. tid={md.get('transcript_id')} chunk={md.get('chunk_index')} dist={r.get('distance')}")
