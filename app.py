import os
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

st.set_page_config(page_title="AI Customer Support RAG", layout="wide")
st.title("AI Customer Support Assistant — RAG Demo")
st.caption("Upload documents, ask questions, and get grounded answers with source snippets.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Add OPENAI_API_KEY to your .env file or Streamlit secrets.")

client = OpenAI(api_key=api_key) if api_key else None


def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def read_pdf(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_sample_docs() -> List[Dict[str, str]]:
    docs = []
    sample_dir = Path("sample_docs")
    for path in sample_dir.glob("*.txt"):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def chunk_text(text: str, source: str) -> List[Dict[str, str]]:
    clean_text = " ".join(text.split())
    chunks = []
    start = 0
    chunk_id = 1
    while start < len(clean_text):
        end = start + CHUNK_SIZE
        chunk = clean_text[start:end]
        if chunk.strip():
            chunks.append({
                "source": source,
                "chunk_id": str(chunk_id),
                "text": chunk,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_id += 1
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def build_index(chunks: List[Dict[str, str]]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    embeddings = embed_texts([c["text"] for c in chunks])
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query: str, index: faiss.IndexFlatIP, chunks: List[Dict[str, str]], top_k: int = 4) -> List[Dict[str, str]]:
    query_embedding = embed_texts([query])
    faiss.normalize_L2(query_embedding)
    scores, ids = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = chunks[idx].copy()
            item["score"] = float(score)
            results.append(item)
    return results


def generate_answer(question: str, contexts: List[Dict[str, str]]) -> str:
    context_block = "\n\n".join(
        [f"Source: {c['source']} | Chunk: {c['chunk_id']}\n{c['text']}" for c in contexts]
    )
    system_prompt = """
You are a helpful AI customer support assistant.
Answer only using the provided context.
If the answer is not in the context, say you do not have enough information.
Be clear, concise, and customer-friendly.
Mention the source document names used.
"""
    user_prompt = f"""
Context:
{context_block}

Customer question:
{question}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


with st.sidebar:
    st.header("Documents")
    use_sample = st.checkbox("Use sample company FAQ", value=True)
    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4)

raw_docs = []
if use_sample:
    raw_docs.extend(load_sample_docs())

for uploaded in uploaded_files or []:
    if uploaded.name.lower().endswith(".pdf"):
        text = read_pdf(uploaded)
    else:
        text = read_txt(uploaded.getvalue())
    raw_docs.append({"source": uploaded.name, "text": text})

if not raw_docs:
    st.info("Upload a document or use the sample FAQ to start.")
    st.stop()

chunks = []
for doc in raw_docs:
    chunks.extend(chunk_text(doc["text"], doc["source"]))

st.write(f"Loaded **{len(raw_docs)}** document(s) and created **{len(chunks)}** chunks.")

if client is None:
    st.stop()

if "rag_index" not in st.session_state or st.button("Rebuild Index"):
    with st.spinner("Creating embeddings and building FAISS index..."):
        index, embeddings = build_index(chunks)
        st.session_state.rag_index = index
        st.session_state.rag_chunks = chunks
    st.success("Index ready.")

question = st.text_input("Ask a customer support question", placeholder="Example: What is the refund policy?")

if question and "rag_index" in st.session_state:
    with st.spinner("Retrieving context and generating answer..."):
        retrieved = retrieve(question, st.session_state.rag_index, st.session_state.rag_chunks, top_k=top_k)
        answer = generate_answer(question, retrieved)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Source snippets")
    for item in retrieved:
        with st.expander(f"{item['source']} — chunk {item['chunk_id']} — score {item['score']:.3f}"):
            st.write(item["text"])
