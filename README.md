# Enterprise RAG Starter (FAISS + OpenAI + Streamlit)

This is a tiny, end-to-end starter you can run in minutes. It:
- Embeds your documents with **OpenAI embeddings**
- Stores vectors in **FAISS**
- Retrieves relevant chunks for a question
- Calls an **LLM** to answer with that context
- Provides a **Streamlit UI**

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here  # Windows PowerShell: $env:OPENAI_API_KEY="your_key_here"
```

*(Optional)* Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you prefer env files.

Put a few **.txt** or **.pdf** files in the `data/` folder (one sample is included).

## 2) Run the UI

```bash
streamlit run streamlit_app.py
```

In the app:
1. Click **Ingest ./data**
2. Ask a question in the input box
3. See the answer + top matching passages

## 3) How it works (4 parts)

1) **Embeddings** — `rag_core.embed_texts()`: text → vectors (OpenAI embeddings)  
2) **FAISS** — `rag_core.ingest_directory()` / `rag_core.retrieve()`: store/search vectors  
3) **Back to text** — we map FAISS IDs → original snippets via `meta.jsonl`  
4) **LLM** — `rag_core.answer()`: build prompt with context → call chat model

## 4) CLI Demo (optional)
You can also run a small CLI demo without Streamlit:
```bash
python simple_cli_demo.py
```

## 5) Files
- `rag_core.py`        — core logic (ingest, retrieve, answer)
- `streamlit_app.py`   — minimal UI
- `simple_cli_demo.py` — quick command-line run
- `data/`              — put your PDFs/TXTs here
- `store/`             — FAISS index + metadata live here (created on ingest)

## Notes
- This starter normalizes vectors and uses `IndexFlatIP` to approximate cosine similarity.
- For real projects, add chunking by tokens, metadata filters, and a reranker.
