"""
ONGC AI Assistant
=================

Flask + FAISS + BAAI/bge-base-en-v1.5 + Ollama llama3.2

Unified retrieval:
  All documents (PDFs from any folder) are searched together as a single
  combined knowledge base. There is no book/report priority split.

Run:
    python app.py

Debug:
    http://127.0.0.1:5050/api/debug?q=TLC
"""

from __future__ import annotations

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["HTTP_PROXY"]           = ""
os.environ["HTTPS_PROXY"]          = ""

import logging
import pickle
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================================================
# PATHS
# =========================================================

BASE_DIR      = Path(__file__).resolve().parent
FAISS_DB_PATH = BASE_DIR / "faiss_db"
PDF_FOLDER    = BASE_DIR / "files"        # Documents
BOOKS_FOLDER  = BASE_DIR / "Books"        # Documents (kept for PDF-serving lookup only)

INDEX_FILE   = FAISS_DB_PATH / "index.faiss"
CHUNKS_FILE  = FAISS_DB_PATH / "chunks.pkl"
PARENTS_FILE = FAISS_DB_PATH / "parents.pkl"
TERMS_FILE   = FAISS_DB_PATH / "terms.pkl"
LOG_FILE     = BASE_DIR / "app_error.log"
CHAT_DB_FILE = BASE_DIR / "chat_history.db"


# =========================================================
# LOGGING
# =========================================================

def configure_logging() -> None:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    handlers: list[logging.Handler] = [console_handler]

    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s")
        )
        handlers.insert(0, file_handler)
    except OSError as exc:
        console_handler.setLevel(logging.INFO)
        logging.getLogger(__name__).warning(
            "Could not open %s for logging: %s", LOG_FILE, exc
        )

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


configure_logging()
log = logging.getLogger("ongc_app")


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# CONFIG
# =========================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM   = 768

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:latest"

CONTEXT_RELEVANCE_THRESHOLD = 12

NOT_FOUND_PHRASES = [
    "not explicitly mentioned",
    "not directly mentioned",
    "not explicitly stated",
    "i cannot find",
    "i don't have information",
    "i do not have information",
    "no information is provided",
    "no information was found",
    "there is no mention",
    "there is no information",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "define", "describe",
    "explain", "for", "from", "give", "how", "in", "is", "me", "model",
    "of", "on", "or", "tell", "the", "to", "what", "with",
}

ACRONYM_EXPANSIONS: dict[str, str] = {
    "MIC":  "microbiologically influenced corrosion microbial microbes biofilm bacteria",
    "TLC":  "top of the line corrosion top-of-the-line corrosion",
    "POTS": "plain old telephone service corrosion pit pitting",
    "CP":   "cathodic protection",
    "AC":   "alternating current interference corrosion",
    "ILI":  "inline inspection in-line inspection",
    "EIS":  "electrochemical impedance spectroscopy",
    "SRB":  "sulfate reducing bacteria sulphate reducing bacteria",
    "H2S":  "hydrogen sulfide sour corrosion",
    "CO2":  "carbon dioxide sweet corrosion",
    "CRA":  "corrosion resistant alloy",
    "HISC": "hydrogen induced stress cracking",
    "SSC":  "sulfide stress cracking",
    "HIC":  "hydrogen induced cracking",
    "SCC":  "stress corrosion cracking",
    "ERW":  "electric resistance welded",
    "API":  "american petroleum institute pipeline standard",
    "NACE": "national association corrosion engineers",
    "FBE":  "fusion bonded epoxy coating",
    "PE":   "polyethylene coating",
    "CML":  "corrosion monitoring location",
    "UTG":  "ultrasonic thickness gauging",
    "MFL":  "magnetic flux leakage",
    "CIPS": "close interval potential survey",
    "DCVG": "direct current voltage gradient",
    "GIS":  "geographic information system pipeline",
    "PIG":  "pig pigging pipeline cleaning inspection",
    "GRE":  "glass reinforced epoxy liner lining",
    "HDPE": "high density polyethylene liner lining",
}


# =========================================================
# STARTUP LOAD
# =========================================================

def load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_embedding_model() -> SentenceTransformer:
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = int(model.get_sentence_embedding_dimension() or 0)
    if dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding model dimension mismatch: expected {EMBEDDING_DIM}, got {dim}"
        )
    log.info("Embedding model ready; dimension=%s", dim)
    return model


def load_database() -> tuple[Any, list, dict, list]:
    required = [INDEX_FILE, CHUNKS_FILE, PARENTS_FILE, TERMS_FILE]
    missing  = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing FAISS database file(s): "
            + ", ".join(missing)
            + ". Run setup_db.py first."
        )

    index    = faiss.read_index(str(INDEX_FILE))
    children = load_pickle(CHUNKS_FILE)
    parents  = load_pickle(PARENTS_FILE)
    terms    = load_pickle(TERMS_FILE)

    if index.d != EMBEDDING_DIM:
        raise RuntimeError(
            f"FAISS index dimension mismatch: expected {EMBEDDING_DIM}, got {index.d}"
        )
    if hasattr(index, "nprobe"):
        index.nprobe = min(32, max(1, getattr(index, "nlist", 32)))

    log.info(
        "Loaded FAISS DB: vectors=%s children=%s parents=%s terms=%s",
        index.ntotal, len(children), len(parents), len(terms),
    )
    return index, children, parents, terms


try:
    embed_model  = load_embedding_model()
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    faiss_index, child_store, parent_store, searchable_terms = load_database()
except Exception as exc:
    log.exception("Startup failed")
    raise SystemExit(f"Startup failed: {exc}") from exc


# =========================================================
# CHAT HISTORY (SQLite)
# =========================================================

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CHAT_DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_cursor():
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_chat_db() -> None:
    with db_cursor() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources         TEXT NOT NULL DEFAULT '[]',
                created_at      TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        """)
    log.info("Chat history DB ready: %s", CHAT_DB_FILE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_title_from_message(message: str) -> str:
    title = " ".join(message.strip().split())
    if len(title) > 50:
        title = title[:47].rstrip() + "..."
    return title or "New Chat"


def create_conversation(first_message: str | None = None) -> dict[str, Any]:
    conv_id = str(uuid.uuid4())
    ts      = now_iso()
    title   = make_title_from_message(first_message) if first_message else "New Chat"
    with db_cursor() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, ts, ts),
        )
    return {"id": conv_id, "title": title, "created_at": ts, "updated_at": ts}


def conversation_exists(conversation_id: str) -> bool:
    with db_cursor() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def touch_conversation(conversation_id: str, title: str | None = None) -> None:
    ts = now_iso()
    with db_cursor() as conn:
        if title:
            conn.execute(
                "UPDATE conversations SET updated_at = ?, title = ? WHERE id = ?",
                (ts, title, conversation_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (ts, conversation_id),
            )


def add_message(conversation_id: str, role: str, content: str, sources: list[str] | None = None) -> None:
    import json as _json
    with db_cursor() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, _json.dumps(sources or []), now_iso()),
        )


def list_conversations() -> list[dict[str, Any]]:
    with db_cursor() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    import json as _json
    with db_cursor() as conn:
        rows = conn.execute(
            "SELECT role, content, sources, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    messages = []
    for r in rows:
        d = dict(r)
        try:
            d["sources"] = _json.loads(d["sources"])
        except (TypeError, ValueError):
            d["sources"] = []
        messages.append(d)
    return messages


def delete_conversation(conversation_id: str) -> bool:
    with db_cursor() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        deleted = cur.rowcount > 0
    return deleted


init_chat_db()


# =========================================================
# HELPERS
# =========================================================

def safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_search_terms(query: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", query)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def exact_phrases(query: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", query).strip().lower()
    words   = [w for w in cleaned.split() if w not in STOPWORDS]
    phrases = []
    if cleaned:
        phrases.append(cleaned)
    if len(words) >= 2:
        phrases.append(" ".join(words))
    return list(dict.fromkeys(phrases))


def query_has_known_acronym(query: str) -> bool:
    for acronym in ACRONYM_EXPANSIONS:
        if re.search(rf"\b{re.escape(acronym)}\b", query, re.IGNORECASE):
            return True
    return False


def expand_query(query: str) -> str:
    additions = []
    for acronym, expansion in ACRONYM_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(acronym)}\b", query, re.IGNORECASE):
            additions.append(expansion)
    return f"{query} {' '.join(additions)}" if additions else query


def decompose_query(query: str) -> list[str]:
    q = query.lower().strip().rstrip("?")
    angles = [query]

    core = re.sub(
        r'^(is|are|can|could|what|how|why|when|does|do|explain|describe|tell me about|'
        r'give me|what is|what are|how does|how do)\s+',
        '', q, flags=re.IGNORECASE
    ).strip()
    if core and core != q:
        angles.append(core)

    extended_stops = STOPWORDS | {
        'that', 'this', 'with', 'from', 'have', 'been', 'will', 'would',
        'could', 'should', 'there', 'their', 'about', 'which', 'when',
        'what', 'possible', 'using', 'used', 'into', 'onto', 'over',
        'under', 'through', 'between', 'during', 'before', 'after',
        'possible', 'internally', 'external', 'generally', 'typically'
    }
    key_terms = [
        w for w in re.findall(r'\b[a-z]{3,}\b', q)
        if w not in extended_stops
    ]

    for i in range(len(key_terms)):
        for j in range(i + 1, min(i + 4, len(key_terms))):
            angles.append(f"{key_terms[i]} {key_terms[j]}")

    angles.extend(key_terms)

    seen, unique = set(), []
    for a in angles:
        a = a.strip()
        if a and a not in seen and len(a) > 2:
            seen.add(a)
            unique.append(a)

    return unique[:10]


# =========================================================
# SPELL CORRECTION
# =========================================================

PROTECTED_TERMS = {
    "iron counts", "iron count", "pigging", "lined riser", "riser",
    "mic", "srb", "tlc", "co2", "h2s", "scc", "lpr", "er probe",
    "feco3", "fes", "corrosion coupon", "electrical resistance probe",
    "linear polarization resistance",
}


def correct_query(query: str) -> str:
    q = query.lower().strip()
    if q in PROTECTED_TERMS:
        return query

    words = query.split()
    corrected = []

    for word in words:
        if len(word) <= 3 or word.isdigit() or word.isupper():
            corrected.append(word)
            continue

        match = process.extractOne(word.lower(), searchable_terms, score_cutoff=90)
        if match:
            candidate, score, _ = match
            corrected.append(candidate if score >= 95 else word)
        else:
            corrected.append(word)

    result = " ".join(corrected)
    if result.lower() != query.lower():
        log.info("Spell corrected: %r -> %r", query, result)
    return result


# =========================================================
# KEYWORD SEARCH  (unified — searches the whole combined database)
# =========================================================

def keyword_search(query: str, max_results: int = 40) -> list[dict]:
    tokens      = extract_search_terms(query)
    caps_tokens = {t.lower() for t in re.findall(r"\b[A-Z]{2,8}\b", query)}
    phrases     = exact_phrases(query)

    scored: list[tuple[int, dict]] = []
    for chunk in child_store:
        text  = safe_text(chunk.get("content")).lower()
        score = 0

        for word in tokens:
            if word in caps_tokens:
                if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
                    score += 200
                continue
            if re.search(rf"\b{re.escape(word)}\b", text):
                score += 10 + len(word)

        for phrase in phrases:
            if len(phrase) >= 3 and phrase in text:
                score += 300 + len(phrase)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:max_results]]


# =========================================================
# SEMANTIC SEARCH  (unified — searches the whole combined database)
# =========================================================

def semantic_search(query: str, top_k: int = 60) -> list[dict]:
    if faiss_index.ntotal == 0:
        return []

    angles  = decompose_query(expand_query(query))
    prefix  = "Represent this sentence for searching relevant passages: "
    queries = [prefix + a for a in angles]

    embeddings = embed_model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    per_k = max(15, top_k // max(1, len(angles)))
    per_k = min(per_k, faiss_index.ntotal)

    seen: set[str] = set()
    results: list[dict] = []

    for emb in embeddings:
        _, indices = faiss_index.search(emb.reshape(1, -1), per_k)
        for raw_idx in indices[0]:
            idx = int(raw_idx)
            if 0 <= idx < len(child_store):
                chunk = child_store[idx]
                content = safe_text(chunk.get("content"))
                if content and content not in seen:
                    seen.add(content)
                    results.append(chunk)

    return results[:top_k]


# =========================================================
# RERANKING
# =========================================================

def rerank_chunks(query: str, chunks: list[dict], top_k: int = 15) -> list[dict]:
    if not chunks:
        return []
    pairs  = [(query, safe_text(c.get("content"))) for c in chunks]
    scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# =========================================================
# HYBRID SEARCH  (unified across the whole combined database)
# =========================================================

def hybrid_search(query: str) -> list[dict]:
    corrected = correct_query(query)

    kw_orig = keyword_search(query,     max_results=30)
    kw_corr = keyword_search(corrected, max_results=30)
    sm_orig = semantic_search(query,     top_k=40)
    sm_corr = semantic_search(corrected, top_k=40)

    seen: set[str] = set()
    combined: list[dict] = []
    for result in kw_orig + kw_corr + sm_orig + sm_corr:
        content = safe_text(result.get("content"))
        if content and content not in seen:
            combined.append(result)
            seen.add(content)

    return rerank_chunks(query, combined, top_k=12)


# =========================================================
# PARENT LOOKUP
# =========================================================

def get_parent_chunks(child_chunks: list[dict]) -> list[dict]:
    seen_parents: dict[str, dict] = {}
    for child in child_chunks:
        parent_id = safe_text(child.get("parent_id"))
        if parent_id and parent_id not in seen_parents:
            parent = parent_store.get(parent_id)
            if parent:
                seen_parents[parent_id] = parent
    return list(seen_parents.values())


# =========================================================
# CONTEXT RELEVANCE GATE
# =========================================================

def context_relevance_score(query: str, parent_chunks: list[dict]) -> float:
    if not parent_chunks:
        return 0.0
    expanded = expand_query(query)
    terms    = extract_search_terms(expanded)
    phrases  = exact_phrases(expanded)
    angles   = decompose_query(query)

    best = 0.0
    for parent in parent_chunks:
        text = safe_text(parent.get("content")).lower()

        fuzzy_full   = fuzz.token_set_ratio(" ".join(terms) or query.lower(), text)
        fuzzy_angle  = max(fuzz.partial_ratio(a.lower(), text) for a in angles)
        phrase_bonus = 45 if any(p in text for p in phrases if len(p) >= 3) else 0
        term_hits    = sum(1 for t in terms if re.search(rf"\b{re.escape(t)}\b", text))
        term_score   = min(45, term_hits * 12)
        exact_hit    = any(
            re.search(rf"\b{re.escape(t)}\b", text)
            for t in extract_search_terms(query)
            if len(t) >= 5
        )
        exact_bonus = 50 if exact_hit else 0

        best = max(best, fuzzy_full, fuzzy_angle, phrase_bonus + term_score + exact_bonus)

    return best


def build_context(parent_chunks: list[dict]) -> str:
    parts = []
    for parent in parent_chunks:
        start = safe_int(parent.get("start_page"))
        end   = safe_int(parent.get("end_page"))
        if start and end and start != end:
            page_tag = f" | pages {start}-{end}"
        elif start:
            page_tag = f" | page {start}"
        else:
            page_tag = ""

        content = safe_text(parent.get("content")).strip()[:2500]
        parts.append(
            f"[SOURCE: {safe_text(parent.get('source'))}{page_tag}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


# =========================================================
# HALLUCINATION GUARD
# =========================================================

NOT_FOUND_RESPONSE = "I could not find this information in the uploaded PDFs."


def _answer_has_substance(answer: str, min_words: int = 12) -> bool:
    return len([w for w in answer.split() if len(w) > 2]) >= min_words


def sanitize_answer(answer: str) -> str:
    lower = answer.lower()

    for phrase in NOT_FOUND_PHRASES:
        if phrase in lower:
            return NOT_FOUND_RESPONSE

    not_found_patterns = [
        r"could not (find|locate|identify)",
        r"(no|not).{0,20}information.{0,30}(pdf|document|context|provided)",
    ]
    for pattern in not_found_patterns:
        if re.search(pattern, lower):
            return NOT_FOUND_RESPONSE

    cleaned = answer.strip()
    if not _answer_has_substance(cleaned):
        return NOT_FOUND_RESPONSE

    cleaned = re.sub(
        r"\s*\[(FULL COVERAGE|PARTIAL COVERAGE[^\]]*|NOT IN DOCUMENTS)\]\s*$",
        "", cleaned, flags=re.IGNORECASE,
    ).strip()

    return cleaned


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_TEMPLATE = """You are a technical assistant for ONGC engineers. Answer every question that has relevant information in the CONTEXT — including indirect, comparative, causal, and procedural questions.

════════════════════════════════════════════════════════════════
ABSOLUTE RULE: Use ONLY the CONTEXT below. No outside knowledge.
════════════════════════════════════════════════════════════════

━━━ BEFORE YOU READ THE QUESTION ━━━
The question may be phrased in ANY of these ways — handle ALL of them:
  • Direct:      "What is TLC?" / "Define pigging"
  • Indirect:    "Is pigging possible in internally lined risers?"
  • Causal:      "Why does corrosion occur at the top of a pipeline?"
  • Comparative: "What is the difference between MIC and general corrosion?"
  • Procedural:  "How do we calculate RSI?" / "How is pigging performed?"
  • Vague:       "What prevents scaling?" → find RSI/LSI thresholds in context
  • Acronym:     "Explain SRB" → look for both short form and full form in context
  • Typo:        "dewaard modle" → treat as de Waard model

━━━ RULES ━━━
1. Answer ONLY from the CONTEXT. Never use outside knowledge. Never.
2. Answer BOTH direct AND indirect questions.
3. Read EVERY passage before deciding the answer is absent.
4. For acronyms — search both the short form AND the full form in the context.
5. For indirect questions — identify the underlying topic and answer from relevant passages.
6. FORMAT — keep it concise (roughly 100-200 words total):
   - Start with a 1-2 sentence direct-answer summary in plain prose.
   - Use bullet points for lists of features, factors, types, causes, steps, etc.
   - Use ## subheadings if items fall into 2-4 distinct groups.
   - Use **bold** only for key terms, parameter names, formulas, or thresholds.
   - Do NOT add a summary section or closing offer.
7. Start directly with the answer — NO preambles or meta-commentary.
   WRONG: "According to the context..." / "The document states..."
   RIGHT: "Pigging in internally lined risers requires..." (just state the facts)
8. State exact numbers, formulas, thresholds, and units as given in CONTEXT.
9. If the topic is truly absent after reading all passages, write EXACTLY:
   "I could not find this information in the uploaded PDFs."
10. End your answer with exactly one tag:
    [FULL COVERAGE] / [PARTIAL COVERAGE - found: X | not found: Y] / [NOT IN DOCUMENTS]
11. Combine information from all relevant passages into one concise answer.
12. Do not reproduce document text verbatim. Summarize and synthesize.
13. Each CONTEXT passage begins with [SOURCE: filename.pdf | page X-Y].
    After the coverage tag, on a NEW final line write:
    USED_SOURCES: <comma-separated exact filenames you relied on>
    - List ONLY sources whose content directly contributed to the answer.
    - If no answer found, write: USED_SOURCES: NONE

CONTEXT:
{context}

QUESTION: {question}
"""


def build_user_message(query: str) -> str:
    return (
        "Answer using ONLY the CONTEXT in the system prompt. "
        "Do NOT reference the context, the document, or the system prompt. "
        "Just state the facts directly. "
        "Keep the answer concise (roughly 100-200 words). Use ## subheadings "
        "and short bullet points only if the topic has multiple distinct "
        "aspects; for simple questions just answer in 1-3 sentences of prose. "
        "No closing summary or follow-up offers. "
        "For acronyms search both the short form and full form. "
        "For indirect questions identify the underlying topic and answer it. "
        "If unsure: I could not find this information in the uploaded PDFs.\n\n"
        f"QUESTION: {query}"
    )


def call_ollama(system_prompt: str, user_message: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature":    0.2,
            "num_predict":    600,
            "repeat_penalty": 1.3,
            "top_p":          0.9,
            "top_k":          40,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("Could not connect to Ollama. Run: ollama serve") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Ollama timed out after 120 s.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc

    try:
        return safe_text(data["message"]["content"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Ollama response shape: {data}") from exc


# =========================================================
# ANSWER GENERATION  (single unified database, no source priority)
# =========================================================

def _build_sources(used_sources_raw: list[str],
                   selected_parents: list[dict]) -> list[str]:
    """Resolve model-reported sources against the actual context sent."""
    available: dict[str, str] = {}
    for parent in selected_parents:
        src = parent.get("source")
        if not src:
            continue
        file_name = safe_text(src).split("\\")[-1].split("/")[-1]
        available[file_name.lower()] = file_name

    sources: list[str] = []
    seen:    set[str]  = set()

    for raw_name in used_sources_raw:
        if raw_name.upper() == "NONE":
            continue
        cleaned_name = raw_name.strip().strip(".").strip("\"'")
        file_name    = cleaned_name.split("\\")[-1].split("/")[-1]
        match_key    = file_name.lower()

        if match_key in available:
            resolved = available[match_key]
            if resolved not in seen:
                sources.append(resolved)
                seen.add(resolved)
            continue

        best = process.extractOne(match_key, list(available.keys()), score_cutoff=85)
        if best:
            resolved = available[best[0]]
            if resolved not in seen:
                sources.append(resolved)
                seen.add(resolved)

    # Fallback if model gave no usable USED_SOURCES
    if not sources:
        for parent in selected_parents[:3]:
            src = parent.get("source")
            if not src:
                continue
            file_name = safe_text(src).split("\\")[-1].split("/")[-1]
            if file_name not in seen:
                sources.append(file_name)
                seen.add(file_name)

    return sources


def generate_answer(query: str) -> dict[str, Any]:
    """
    Unified retrieval: search the entire combined database in one pass,
    no book/report split or priority.
    """
    children = hybrid_search(query)
    parents  = get_parent_chunks(children)
    score    = context_relevance_score(query, parents)

    log.info("Relevance score=%.2f  query=%r", score, query)

    if score < 3 or not parents:
        log.info("No relevant content found in the database.")
        return {"answer": NOT_FOUND_RESPONSE, "sources": []}

    selected_parents = parents[:8]

    context       = build_context(selected_parents)
    system_prompt = SYSTEM_TEMPLATE.format(context=context, question=query)
    user_message  = build_user_message(query)

    raw     = call_ollama(system_prompt, user_message)
    cleaned = re.sub(r"\n{3,}", "\n\n", raw).strip()

    # Extract USED_SOURCES
    used_sources_raw: list[str] = []
    m = re.search(r"USED_SOURCES:\s*(.+)\s*$", cleaned,
                  flags=re.IGNORECASE | re.MULTILINE)
    if m:
        used_sources_raw = [s.strip() for s in m.group(1).split(",") if s.strip()]
        cleaned = re.sub(r"\n?USED_SOURCES:.*\s*$", "", cleaned,
                         flags=re.IGNORECASE | re.MULTILINE).strip()

    answer = sanitize_answer(cleaned)

    sources: list[str] = []
    if answer != NOT_FOUND_RESPONSE:
        sources = _build_sources(used_sources_raw, selected_parents)

    log.info("Answer generated | sources=%s", sources)

    return {"answer": answer, "sources": sources}


# =========================================================
# DEBUG PAYLOAD
# =========================================================

def build_debug_payload(query: str) -> dict[str, Any]:
    corrected = correct_query(query)
    expanded  = expand_query(corrected)
    angles    = decompose_query(query)

    children = hybrid_search(query)
    parents  = get_parent_chunks(children)
    score    = context_relevance_score(query, parents)

    payload = {
        "original_query":    query,
        "corrected_query":   corrected,
        "expanded_query":    expanded,
        "decomposed_angles": angles,
        "relevance_score":   score,
        "relevance_threshold": 3,
        "top_chunks": [
            {
                "source":  safe_text(c.get("source")),
                "page":    safe_int(c.get("page_num")),
                "preview": safe_text(c.get("content"))[:300],
            }
            for c in children[:8]
        ],
    }
    return normalize_for_json(payload)


# =========================================================
# ROUTES
# =========================================================

@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    log.exception("Unhandled Flask error on %s %s", request.method, request.path)
    return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/")
def home():
    return send_file(BASE_DIR / "index.html")


@app.route("/ongc_logo.png")
def logo():
    return send_file(BASE_DIR / "ongc_logo.png")


@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    import urllib.parse
    safe_name = Path(urllib.parse.unquote(filename)).name
    # Search all known document folders
    for folder in [PDF_FOLDER, BOOKS_FOLDER]:
        pdf_path = folder / safe_name
        if pdf_path.exists():
            return send_file(str(pdf_path), mimetype="application/pdf")
    return jsonify({"error": f"PDF not found: {safe_name}"}), 404


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(silent=True) or {}
        message = safe_text(data.get("message")).strip()
        conversation_id = safe_text(data.get("conversation_id")).strip() or None

        if not message:
            return jsonify({"error": "Message required"}), 400

        if not conversation_id or not conversation_exists(conversation_id):
            conv = create_conversation(first_message=message)
            conversation_id = conv["id"]
        else:
            touch_conversation(conversation_id)

        add_message(conversation_id, "user", message)
        result = generate_answer(message)
        add_message(conversation_id, "assistant", result["answer"], result["sources"])

        return jsonify(normalize_for_json({
            "message":         result["answer"],
            "sources":         result["sources"],
            "conversation_id": conversation_id,
        }))
    except RuntimeError as exc:
        log.exception("Runtime error in /api/chat")
        return jsonify({"message": str(exc), "sources": []}), 200
    except Exception as exc:
        log.exception("Failed /api/chat")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    try:
        return jsonify({"conversations": list_conversations()})
    except Exception as exc:
        log.exception("Failed /api/conversations")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/api/conversations", methods=["POST"])
def post_new_conversation():
    try:
        conv = create_conversation()
        return jsonify(conv), 201
    except Exception as exc:
        log.exception("Failed to create conversation")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/api/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    try:
        if not conversation_exists(conversation_id):
            return jsonify({"error": "Conversation not found"}), 404
        messages = get_conversation_messages(conversation_id)
        return jsonify({"id": conversation_id, "messages": messages})
    except Exception as exc:
        log.exception("Failed /api/conversations/<id>")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation_route(conversation_id):
    try:
        deleted = delete_conversation(conversation_id)
        if not deleted:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"status": "deleted", "id": conversation_id})
    except Exception as exc:
        log.exception("Failed to delete conversation")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/api/debug", methods=["GET"])
def debug_retrieval():
    try:
        query = safe_text(request.args.get("q")).strip()
        if not query:
            return jsonify({"error": "Pass ?q=your+query"}), 400
        return jsonify(build_debug_payload(query))
    except Exception as exc:
        log.exception("Failed /api/debug")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "embedding_model": EMBEDDING_MODEL,
        "faiss_vectors":   int(faiss_index.ntotal),
        "child_chunks":    len(child_store),
        "parent_chunks":   len(parent_store),
        "ollama_model":    OLLAMA_MODEL,
    })


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  ONGC AI Assistant")
    print(f"  Embeddings : {EMBEDDING_MODEL}")
    print(f"  FAISS      : dim={faiss_index.d}, vectors={faiss_index.ntotal}")
    print(f"  Documents  : {len(child_store):,} chunks (unified database)")
    print(f"  Ollama     : {OLLAMA_MODEL}")
    print("  URL        : http://127.0.0.1:5050")
    print("  Debug      : http://127.0.0.1:5050/api/debug?q=TLC")
    print("=" * 56 + "\n")

    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)