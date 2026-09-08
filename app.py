import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ============================================================
# IIUI Policy Assistant
# RAG + FAISS + Sentence Transformers + Groq + Streamlit
# ============================================================

APP_TITLE = "IIUI Policy Assistant"

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.pkl"

# Multilingual embedding model: useful for English and Urdu queries.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Groq model. You can change this later from the Streamlit sidebar.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# -----------------------------
# Models / API
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    """Read the Groq key from Streamlit Secrets or environment variables."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# PDF ingestion
# -----------------------------

def extract_pdf_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                }
            )

    return pages


def split_text(text, chunk_size=1000, overlap=150):
    """Simple chunking suitable for a beginner RAG project."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks_from_pdfs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        return []

    all_chunks = []

    for pdf_file in pdf_files:
        for page_info in extract_pdf_pages(pdf_file):
            page_chunks = split_text(page_info["text"])

            for chunk_number, chunk in enumerate(page_chunks, start=1):
                all_chunks.append(
                    {
                        "text": chunk,
                        "source": page_info["source"],
                        "page": page_info["page"],
                        "chunk": chunk_number,
                    }
                )

    return all_chunks


# -----------------------------
# FAISS vector index
# -----------------------------

def create_faiss_index():
    chunks = build_chunks_from_pdfs()

    if not chunks:
        raise ValueError(
            "No readable PDF files were found in the data folder."
        )

    model = load_embedding_model()
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # With normalized embeddings, inner product approximates cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    with open(CHUNKS_FILE, "wb") as file:
        pickle.dump(chunks, file)

    return len(chunks)


@st.cache_resource
def load_index_and_chunks():
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        return None, None

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def search_policy(question, top_k=5):
    index, chunks = load_index_and_chunks()

    if index is None or chunks is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, positions = index.search(query_embedding, top_k)

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position == -1:
            continue

        result = chunks[position].copy()
        result["score"] = float(score)
        results.append(result)

    return results


# -----------------------------
# RAG prompt + Groq
# -----------------------------

def make_context(results):
    parts = []

    for i, result in enumerate(results, start=1):
        parts.append(
            f"[SOURCE {i}]\n"
            f"Document: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Content:\n{result['text']}"
        )

    return "\n\n".join(parts)


def ask_groq(question, results, model_name):
    client = get_groq_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    context = make_context(results)

    system_prompt = """
You are the IIUI Policy Assistant.

Your job is to answer questions ONLY using the policy context supplied
to you.

Strict rules:
1. Do not invent IIUI rules, procedures, dates, fees, penalties,
   requirements, or exceptions.
2. If the supplied context does not contain enough information,
   clearly say that the answer was not found in the available IIUI
   policy documents.
3. Give a concise, clear answer suitable for students, faculty,
   and university staff.
4. Whenever possible, cite the relevant document name and page number.
5. If different retrieved passages appear to conflict, do not decide
   which one is correct. Explain the conflict and advise checking the
   latest official IIUI policy.
6. The official IIUI policy and competent university authority remain
   the final authority.
7. You may answer in the same language as the user's question.
"""

    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED IIUI POLICY CONTEXT:
{context}

Using only the retrieved context, answer the user's question.
At the end, provide a short "Sources" section listing the relevant
document names and page numbers.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_completion_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Interface
# -----------------------------

st.title("🎓 IIUI Policy Assistant")
st.caption(
    "RAG-based university policy assistant using Sentence Transformers, "
    "FAISS and Groq"
)

with st.sidebar:
    st.header("⚙️ Settings")

    groq_model = st.selectbox(
        "Groq model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        index=0,
    )

    top_k = st.slider(
        "Retrieved policy passages",
        min_value=2,
        max_value=8,
        value=5,
    )

    st.divider()

    st.subheader("📚 Policy Index")

    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Reading policy PDFs and creating FAISS index..."):
            try:
                count = create_faiss_index()
                load_index_and_chunks.clear()
                st.success(f"Index created: {count} text chunks.")
            except Exception as error:
                st.error(f"Indexing failed: {error}")

    st.divider()

    if get_groq_client():
        st.success("Groq API key detected.")
    else:
        st.warning("Groq API key not detected.")

    st.info(
        "For deployment, store GROQ_API_KEY in Streamlit Secrets. "
        "Never put the key directly into app.py or GitHub."
    )


if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
    st.warning(
        "No policy index found. Put IIUI policy PDFs in the `data` folder "
        "and click **Build / Rebuild Index**."
    )


# -----------------------------
# Chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about IIUI policies..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching IIUI policies and generating answer..."):
            try:
                results = search_policy(question, top_k=top_k)

                if not results:
                    answer = (
                        "I could not find relevant information in the "
                        "indexed IIUI policy documents. Please check that "
                        "the correct policy PDFs have been uploaded and "
                        "indexed."
                    )
                else:
                    answer = ask_groq(
                        question,
                        results,
                        groq_model,
                    )

                st.markdown(answer)

                with st.expander("🔎 Retrieved policy passages"):
                    for i, result in enumerate(results, start=1):
                        st.markdown(
                            f"**{i}. {result['source']} — "
                            f"Page {result['page']} — "
                            f"Similarity {result['score']:.3f}**"
                        )
                        st.write(result["text"])

            except Exception as error:
                answer = (
                    f"An error occurred: `{error}`\n\n"
                    "Check your Groq API key, model name, internet "
                    "connection, and policy index."
                )
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


st.divider()

st.caption(
    "⚠️ This assistant provides information from indexed IIUI policy "
    "documents. Always verify important administrative decisions against "
    "the latest official IIUI policy and the competent authority."
)
