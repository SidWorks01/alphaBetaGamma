import json
import os
import threading
import uuid
import tempfile
from typing import Any, Dict, TypedDict, Optional

from langgraph.graph import StateGraph

PERSIST_FILE = "data/persistent_dirs.json"
# Use a re-entrant lock to allow the same thread to re-acquire if needed,
# but we also avoid nested locking by design below.
_PERSIST_LOCK = threading.RLock()


# ---------- Helpers ----------
def _new_id() -> str:
    return str(uuid.uuid4())


# ---------- Persistent JSON Helpers ----------
def _init_persist_if_missing():
    """
    Ensure the persistent file exists and contains a JSON list.
    NOTE: do not acquire _PERSIST_LOCK here to avoid nested-lock patterns.
    The caller (create_new_entry / update) will hold the lock when needed.
    """
    if not os.path.exists(PERSIST_FILE):
        # create file with empty list (no lock here)
        with open(PERSIST_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def load_all_entries() -> list:
    """
    Load the entire JSON list from disk.
    This function does NOT acquire the lock — callers that need consistent view
    should call it while holding _PERSIST_LOCK.
    """
    _init_persist_if_missing()
    with open(PERSIST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(obj: Any, filename: str):
    """
    Atomically write JSON to filename by writing to a temp file and os.replace.
    Caller should hold _PERSIST_LOCK for correctness across threads.
    """
    dirpath = os.path.dirname(os.path.abspath(filename)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp_persist_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            json.dump(obj, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        # atomic replace
        os.replace(tmp_path, filename)
    finally:
        # cleanup if temp still exists
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def save_all_entries(entries: list):
    """
    Save the entire list to the persistent JSON file atomically.
    This function does NOT acquire the lock — callers should hold _PERSIST_LOCK.
    """
    _atomic_write_json(entries, PERSIST_FILE)


def create_new_entry(query: str) -> str:
    """
    Create a new directory entry for this query.
    Structure:
    {
      "id": "<uuid>",
      "query": "<string>",
      "node3_output": "",
      "node4_output": ""
    }
    This function acquires _PERSIST_LOCK so concurrent create/update calls are safe.
    """
    _init_persist_if_missing()
    entry = {
        "id": _new_id(),
        "query": query,
        "node3_output": "",
        "node4_output": ""
    }

    with _PERSIST_LOCK:
        # read-modify-write while holding lock
        entries = load_all_entries()
        entries.append(entry)
        save_all_entries(entries)

    return entry["id"]


def update_entry_output_string(entry_id: str, key: str, value_str: str) -> bool:
    """
    Update "node3_output" or "node4_output" (store plain string).
    Uses a single lock-protected read-modify-write cycle (no nested locks).
    Returns True if updated, False if entry not found.
    """
    if key not in ("node3_output", "node4_output"):
        raise ValueError("Invalid key. Allowed: node3_output, node4_output")

    with _PERSIST_LOCK:
        entries = load_all_entries()
        found = False
        for e in entries:
            if e.get("id") == entry_id:
                e[key] = value_str
                found = True
                break
        if found:
            save_all_entries(entries)
            return True
        else:
            return False


# ---------- LangGraph State ----------
class MyState(TypedDict, total=False):
    # start: stores only the query string itself (per your clarification)
    query: str
    persistent_entry_id: str

    node1_output: Any
    node2_output: Any

    node3_input_snapshot: Any
    node4_input_snapshot: Any
    node3_output: Any
    node4_output: Any

    node5_output: Any
    node6_output: Any
    final_eval_result: Any


# ---------- Node Definitions ----------
def start_fn(state: MyState) -> Dict[str, Any]:
    """
    Produces the query string and creates a new persistent entry.
    (Example query here; in real usage supply the real string)
    """
    # Replace this with actual query input as required
    query = "search: find top-10 widgets matching 'fast' sorted by price"
    state["query"] = query

    entry_id = create_new_entry(query)
    state["persistent_entry_id"] = entry_id

    return {"query": query, "persistent_entry_id": entry_id}


def node1_fn(state: MyState) -> Dict[str, Any]:
    processed = f"node1 processed query: {state['query']}"
    state["node1_output"] = {"node1": processed}
    return {"node1_output": state["node1_output"]}


def node2_fn(state: MyState) -> Dict[str, Any]:
    payload = f"payload derived from node1: {state['node1_output']}"
    state["node2_output"] = {"payload": payload}
    return {"node2_output": state["node2_output"]}


def node3_fn(state: MyState) -> Dict[str, Any]:
    """
    Node3 snapshots input into state, produces a string output, then updates persistent entry.
    This will acquire the lock only once for the update call.
    """
    input_snapshot = state.get("node2_output")
    state["node3_input_snapshot"] = input_snapshot

    out_str = f"node3 processed payload: {input_snapshot}"
    state["node3_output"] = out_str

    # Update persistent JSON (store as plain string)
    entry_id: Optional[str] = state.get("persistent_entry_id")
    if entry_id:
        # single lock-protected call
        ok = update_entry_output_string(entry_id, "node3_output", out_str)
        if not ok:
            # optional: handle missing entry (log, raise, etc.). Here we just continue.
            pass

    return {"node3_output": out_str}


def node4_fn(state: MyState) -> Dict[str, Any]:
    """
    Node4 snapshots input into state, produces a string output, then updates the persistent entry.
    """
    input_snapshot = state.get("node2_output")
    state["node4_input_snapshot"] = input_snapshot

    out_str = f"node4 processed payload: {input_snapshot}"
    state["node4_output"] = out_str

    entry_id: Optional[str] = state.get("persistent_entry_id")
    if entry_id:
        ok = update_entry_output_string(entry_id, "node4_output", out_str)
        if not ok:
            pass

    return {"node4_output": out_str}


def node5_fn(state: MyState) -> Dict[str, Any]:
    combined = f"combined node3/node4 -> {state.get('node3_output')} | {state.get('node4_output')}"
    state["node5_output"] = combined
    return {"node5_output": combined}


def node6_fn(state: MyState) -> Dict[str, Any]:
    out = f"node6 uses node5: {state.get('node5_output')}"
    state["node6_output"] = out
    return {"node6_output": out}


def eval_fn(state: MyState) -> Dict[str, Any]:
    result = f"final eval based on node6: {state.get('node6_output')}"
    state["final_eval_result"] = result
    # per your instruction, DO NOT persist final output
    return {"final_eval_result": result}


# ---------- Build Graph ----------
graph = StateGraph(MyState)

graph.add_node("start", start_fn)
graph.add_node("node1", node1_fn)
graph.add_node("node2", node2_fn)
graph.add_node("node3", node3_fn)
graph.add_node("node4", node4_fn)
graph.add_node("node5", node5_fn)
graph.add_node("node6", node6_fn)
graph.add_node("eval", eval_fn)

graph.set_entry_point("start")
graph.set_finish_point("eval")

graph.add_edge("start", "node1")
graph.add_edge("node1", "node2")
graph.add_edge("node2", "node3")
graph.add_edge("node2", "node4")
graph.add_edge("node3", "node5")
graph.add_edge("node4", "node5")
graph.add_edge("node5", "node6")
graph.add_edge("node6", "eval")


# ---------- Run ----------
if __name__ == "__main__":
    # Ensure persistent file exists (no lock held here; functions that write will lock)
    _init_persist_if_missing()

    compiled = graph.compile()
    final_state = compiled.invoke({})

    # Print final in-memory state
    print("\nFinal in-memory state:")
    for key in ["query", "persistent_entry_id", "node3_output", "node4_output", "final_eval_result"]:
        print(f"{key}: {final_state.get(key)}")

    # Print persistent JSON content
    print("\nPersistent JSON content:")
    with open(PERSIST_FILE, "r", encoding="utf-8") as f:
        print(f.read())
