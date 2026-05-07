# AI Customer Support Assistant (LLM + RAG)

A Streamlit app that answers customer questions from uploaded or sample documents using Retrieval-Augmented Generation (RAG), OpenAI embeddings, and FAISS vector search.

## Resume Keywords Covered
- LLMs
- Prompt Engineering
- RAG
- Embeddings
- Vector Search
- LangChain-style workflow
- AI response evaluation
- Customer-facing AI use case

## Features
- Upload TXT or PDF documents
- Build local FAISS vector index
- Retrieve relevant chunks
- Generate grounded GPT answers
- Show source snippets for transparency

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`.

## Run

```bash
streamlit run app.py
```

## Deploy

### Streamlit Community Cloud
1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app and select `app.py`.
4. Add `OPENAI_API_KEY` in app secrets.
5. Deploy.

## Example Resume Bullet
Built a RAG-based AI customer support assistant using OpenAI GPT, embeddings, and FAISS vector search to answer customer questions from internal documents, improving answer relevance and reducing hallucinations through prompt engineering and source-grounded responses.
