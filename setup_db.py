"""
ONGC AI Assistant - Database Setup (FAISS)
==========================================
  ✓ bge-base-en-v1.5 embeddings (fast CPU, 768-dim)
  ✓ Parent-child chunking (child → FAISS, parent → rich LLM context)
  ✓ Page-number metadata stored per chunk
  ✓ Persistent FAISS storage with incremental loading
  ✓ Automatic IVFFlat upgrade for large corpora (>50 000 vectors)
  ✓ Dynamic spell-correction vocabulary

Usage:
    pip install faiss-cpu pypdf sentence-transformers
    python setup_db.py
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
import re
import json
import pickle
import numpy as np
import pypdf
import faiss

from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer

# =========================================================
# PATH CONFIG
# =========================================================

FAISS_DB_PATH = "./faiss_db"
PDF_FOLDER    = "./files"

INDEX_FILE   = f"{FAISS_DB_PATH}/index.faiss"
CHUNKS_FILE  = f"{FAISS_DB_PATH}/chunks.pkl"
PARENTS_FILE = f"{FAISS_DB_PATH}/parents.pkl"
TRACKER_FILE = f"{FAISS_DB_PATH}/loaded_pdfs.json"
TERMS_FILE   = f"{FAISS_DB_PATH}/terms.pkl"

os.makedirs(FAISS_DB_PATH, exist_ok=True)

# =========================================================
# CHUNKING PARAMETERS
# =========================================================

CHILD_SIZE     = 512    # characters — what FAISS indexes (precise match)
CHILD_OVERLAP  = 100
PARENT_SIZE    = 1800   # characters — what LLM reads (rich context)
PARENT_OVERLAP = 200

# =========================================================
# EMBEDDING MODEL  (bge-base-en-v1.5 — fast on CPU)
# =========================================================

print("Loading embedding model: BAAI/bge-base-en-v1.5 ...")
embed_model   = SentenceTransformer("BAAI/bge-base-en-v1.5")
EMBEDDING_DIM = 768
print("Model ready.\n")

# =========================================================
# TRACKER
# =========================================================

def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_tracker(tracker):
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)

# =========================================================
# FAISS INDEX
# =========================================================

def load_or_create_index():
    if (os.path.exists(INDEX_FILE)
            and os.path.exists(CHUNKS_FILE)
            and os.path.exists(PARENTS_FILE)):
        print("Loading existing FAISS index...")
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            child_store = pickle.load(f)
        with open(PARENTS_FILE, "rb") as f:
            parent_store = pickle.load(f)
        print(f"Loaded {index.ntotal:,} vectors | "
              f"{len(child_store):,} child chunks | "
              f"{len(parent_store):,} parent chunks\n")
        return index, child_store, parent_store

    print(f"Creating new FAISS index (IndexFlatIP, dim={EMBEDDING_DIM})...\n")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    return index, [], {}

def maybe_upgrade_index(index):
    """Auto-upgrade FlatIP → IVFFlat when corpus grows large."""
    if isinstance(index, faiss.IndexFlatIP) and index.ntotal > 50_000:
        print("Upgrading FAISS index to IVFFlat for speed...")
        nlist     = 256
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        new_index = faiss.IndexIVFFlat(
            quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT
        )
        existing = np.zeros((index.ntotal, EMBEDDING_DIM), dtype="float32")
        index.reconstruct_n(0, index.ntotal, existing)
        new_index.train(existing)
        new_index.add(existing)
        new_index.nprobe = 32
        print("IVFFlat upgrade complete.\n")
        return new_index
    return index

def save_index(index, child_store, parent_store):
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(child_store, f)
    with open(PARENTS_FILE, "wb") as f:
        pickle.dump(parent_store, f)

# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pages(pdf_path):
    pages = []
    try:
        with open(pdf_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
        print(f"   Extracted {len(pages)} pages with text "
              f"(of {len(reader.pages)} total)")
    except Exception as e:
        print(f"   PDF error: {e}")
    return pages

# =========================================================
# PARENT-CHILD CHUNKING
# =========================================================

def make_chunks(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += size - overlap
    return chunks

def build_parent_child(pages, pdf_name):
    full_text   = ""
    page_breaks = []
    for page_num, text in pages:
        page_breaks.append((len(full_text), page_num))
        full_text += text + "\n"

    def char_to_page(char_idx):
        page_num = page_breaks[0][1]
        for offset, pnum in page_breaks:
            if char_idx >= offset:
                page_num = pnum
            else:
                break
        return page_num

    # Build parents
    parents_dict = {}
    parent_texts = make_chunks(full_text, PARENT_SIZE, PARENT_OVERLAP)
    char_cursor  = 0

    for p_idx, p_text in enumerate(parent_texts):
        p_start     = full_text.find(p_text, char_cursor)
        p_end       = p_start + len(p_text) if p_start != -1 else char_cursor
        char_cursor = max(char_cursor, p_end - PARENT_OVERLAP)

        parent_id = f"{pdf_name}::parent_{p_idx}"
        parents_dict[parent_id] = {
            "content":    p_text,
            "source":     pdf_name,
            "start_page": char_to_page(max(0, p_start)),
            "end_page":   char_to_page(max(0, p_end - 1)),
        }

    # Build children from each parent
    children = []
    for parent_id, parent in parents_dict.items():
        child_texts = make_chunks(parent["content"], CHILD_SIZE, CHILD_OVERLAP)
        for c_idx, c_text in enumerate(child_texts):
            c_pos    = full_text.find(c_text[:80])
            page_num = char_to_page(c_pos) if c_pos != -1 else parent["start_page"]
            children.append({
    "content": c_text,
    "parent_id": parent_id,
    "source": pdf_name,
    "page_num": page_num,
    "chunk_id": c_idx,
    "parent_content": parents_dict[parent_id]["content"]  # ADD THIS
})

    return parents_dict, children

# =========================================================
# EMBEDDING  (bge-base-en-v1.5)
# =========================================================

def embed_texts(texts):
    arr = embed_model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    faiss.normalize_L2(arr)
    return arr

# =========================================================
# MAIN
# =========================================================

def load_pdfs_to_database():
    tracker                          = load_tracker()
    index, child_store, parent_store = load_or_create_index()

    print("=" * 60)
    print("  ONGC Vector Database Setup  (bge-base + Parent-Child)")
    print("=" * 60)

    pdf_files = list(Path(PDF_FOLDER).rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in: {os.path.abspath(PDF_FOLDER)}")
        return

    print(f"Found {len(pdf_files)} PDF(s)\n")

    new_loaded = 0
    all_terms  = Counter()

    for pdf_file in pdf_files:
        name = pdf_file.name

        if name in tracker:
            print(f"Skipping already loaded: {name}")
            continue

        print(f"\nLoading: {name}")

        pages = extract_pages(str(pdf_file))
        if not pages:
            print("   Empty / unreadable — skipping.\n")
            continue

        parents_dict, children = build_parent_child(pages, name)
        print(f"   Built {len(parents_dict):,} parents | {len(children):,} children")

        if not children:
            print("   No chunks produced — skipping.\n")
            continue

        # Embed child chunks
        child_texts = [c["content"] for c in children]
        child_emb   = embed_texts(child_texts)
        print(f"   Embedded {len(child_texts):,} child chunks (shape {child_emb.shape})")

        # Add to FAISS
        index.add(child_emb)
        index = maybe_upgrade_index(index)

        # Update stores
        child_store.extend(children)
        parent_store.update(parents_dict)

        # Build vocabulary
        for parent in parents_dict.values():
            for word in re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]+\b', parent["content"]):
                w = word.lower()
                if len(w) > 3:
                    all_terms[w] += 1

        tracker[name] = {
            "parents":  len(parents_dict),
            "children": len(children),
            "status":   "loaded",
        }
        new_loaded += 1

        # Checkpoint after every PDF
        save_index(index, child_store, parent_store)
        save_tracker(tracker)
        print(f"   Checkpoint saved.")

    # Save vocabulary
    filtered_terms = [t for t, cnt in all_terms.items() if cnt >= 2]
    with open(TERMS_FILE, "wb") as f:
        pickle.dump(filtered_terms, f)
    print(f"\nSaved {len(filtered_terms):,} vocabulary terms.")

    print("\n" + "=" * 60)
    if new_loaded == 0:
        print("All PDFs already loaded. Nothing new to process.")
    else:
        print(f"{new_loaded} PDF(s) loaded successfully.")
    print(f"Total child vectors : {index.ntotal:,}")
    print(f"Total parent chunks : {len(parent_store):,}")
    print("=" * 60)
    print("\nNow run:  python app.py\n")

if __name__ == "__main__":
    load_pdfs_to_database()