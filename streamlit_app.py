import streamlit as st
from rag_core import ingest_directory, answer

st.set_page_config(page_title="Enterprise RAG Starter", page_icon="🤖", layout="centered")

st.title("🤖 Enterprise RAG Starter (FAISS + OpenAI)")
st.write("Ingest documents from ./data, then ask questions. The bot answers strictly from your docs.")

with st.expander("How it works"):
    st.markdown("""
    1) **Embeddings**: your docs and question are converted to vectors (numbers).  
    2) **FAISS**: vectors are stored/searched by meaning.  
    3) **Back to text**: we map results to original snippets.  
    4) **LLM**: generates an answer using those snippets.  
    """)

col1, col2 = st.columns(2)
with col1:
    if st.button("📥 Ingest ./data"):
        try:
            n = ingest_directory("data")
            st.success(f"Ingested {n} chunks. You can ask questions now.")
        except Exception as e:
            st.error(f"Ingest failed: {e}")
with col2:
    st.markdown("[Open data folder](.)")  # hint to drop files

query = st.text_input("Ask a question about your documents:", placeholder="What is the expense policy?" )

if st.button("💬 Ask") and query:
    with st.spinner("Thinking..."):
        try:
            result = answer(query, k=5)
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Top passages")
            for p in result["passages"]:
                st.markdown(f"- **{p['source']}** — score: {p['score']:.3f}\n\n> {p['text'][:500]}...")
        except Exception as e:
            st.error(f"Error: {e}")
