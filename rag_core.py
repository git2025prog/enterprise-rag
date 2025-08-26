import os, json, glob
from typing import List, Dict
import numpy as np
import faiss
from openai import OpenAI

# ---------- config ----------
EMBED_MODEL = "text-embedding-3-small"   # 1536 dims
CHAT_MODEL  = "gpt-4o-mini"              # or "gpt-3.5-turbo"
DIMENSIONS  = 1536
INDEX_PATH  = "store/index.faiss"
META_PATH   = "store/meta.jsonl"
DOC_DIR     = "data"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- helpers ----------
def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def embed_texts(texts: List[str]) -> np.ndarray:
    """Call OpenAI embeddings API for a list of strings -> (N, D) float32 array."""
    if not texts:
        return np.zeros((0, DIMENSIONS), dtype="float32")
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype="float32")
    return _normalize(vecs)

def chunk_text(text: str, max_chars: int = 1800) -> List[str]:
    """Very simple chunker by character count (token-aware is better in prod)."""
    text = text.strip()
    if not text:
        return []
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    return [c for c in chunks if c.strip()]

def load_text_from_path(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"[warn] Could not read PDF {path}: {e}")
            return ""
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"[warn] Could not read file {path}: {e}")
            return ""

def _ensure_index(dim: int = DIMENSIONS):
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatIP(dim)  # use inner-product with normalized vectors (cosine-like)

def _append_metadata(records: List[Dict]):
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _read_metadata() -> List[Dict]:
    if not os.path.exists(META_PATH):
        return []
    with open(META_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ---------- 1) INGEST: docs -> chunks -> embeddings -> FAISS + metadata ----------
def ingest_directory(doc_dir: str = DOC_DIR) -> int:
    index = _ensure_index(DIMENSIONS)
    files = [p for p in glob.glob(os.path.join(doc_dir, "**/*"), recursive=True) if os.path.isfile(p)]
    new_records: List[Dict] = []
    all_chunks: List[str] = []

    for path in files:
        text = load_text_from_path(path)
        chunks = chunk_text(text)
        if not chunks:
            continue
        # keep chunks + metadata aligned
        for c in chunks:
            all_chunks.append(c)
            new_records.append({"source": path, "text": c})

    if not all_chunks:
        return 0

    vecs = embed_texts(all_chunks)
    index.add(vecs)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    _append_metadata(new_records)
    return len(new_records)

# ---------- 2) RETRIEVE: question -> embedding -> top-k IDs ----------
def retrieve(query: str, k: int = 5) -> List[Dict]:
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError("Index not found. Run ingest_directory() first.")
    index = faiss.read_index(INDEX_PATH)
    metadata = _read_metadata()
    if not metadata:
        return []

    qv = embed_texts([query])
    D, I = index.search(qv, k)  # similarities & indices
    I = I[0].tolist(); D = D[0].tolist()

    hits = []
    for j, idx in enumerate(I):
        if 0 <= idx < len(metadata):
            m = metadata[idx]
            hits.append({"score": float(D[j]), "source": m["source"], "text": m["text"]})
    return hits

# ---------- 3+4) PROMPT + LLM ANSWER ----------
def answer(query: str, k: int = 5) -> Dict:
    passages = retrieve(query, k=k)
    if not passages:
        prompt = f"You are a helpful assistant. If unknown, say you don't know.\n\nQuestion: {query}\n\nContext: (none)"
    else:
        context_blob = "\n\n---\n\n".join(p['text'] for p in passages)
        prompt = (
            "You answer strictly using the provided context. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context_blob}\n\nQuestion: {query}"
        )

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return {"answer": chat.choices[0].message.content.strip(), "passages": passages}
