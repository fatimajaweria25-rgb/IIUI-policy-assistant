import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# OCR imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# -----------------------------
# 1. SETUP
# -----------------------------
load_dotenv()  # Load OPENAI_API_KEY from .env

st.set_page_config(page_title="IIUI Policy Assistant", page_icon="🎓", layout="wide")

# -----------------------------
# 2. SIDEBAR — ROLE SELECTION
# -----------------------------
st.sidebar.title("🎓 IIUI Policy Assistant")
st.sidebar.markdown("Ask questions about IIUI policies, academic regulations, admissions, fees, etc.")

user_role = st.sidebar.radio(
    "You are:",
    ("Faculty Member", "Student", "Outsider / Prospective Student")
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Knowledge Base")

# Upload documents (with image support for OCR)
uploaded_files = st.sidebar.file_uploader(
    "Upload policy documents (PDF, DOCX, TXT, PNG, JPG)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Add URL
url_input = st.sidebar.text_input("Or add an IIUI URL (e.g., https://www.iiu.edu.pk)")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# 3. ACCESS CONTROL MATRIX
# -----------------------------
ACCESS_MATRIX = {
    "Faculty Member": ["Admissions", "Fees", "Academic", "Examination",
                       "Student Affairs", "Faculty / HR", "Scholarships",
                       "Internship", "Hostel", "Transport", "Hospital",
                       "Departments", "General University", "Public"],
    "Student": ["Admissions", "Fees", "Academic", "Examination",
                "Student Affairs", "Scholarships", "Internship",
                "Hostel", "Transport", "Hospital", "Departments",
                "General University", "Public"],
    "Outsider / Prospective Student": ["Admissions", "Fees", "Public",
                                       "General University", "Scholarships",
                                       "Hostel", "Departments"]
}

# -----------------------------
# 4. OCR FUNCTIONS
# -----------------------------
def ocr_pdf(file_path):
    """Extract text from scanned PDF using OCR."""
    text = ""
    try:
        images = convert_from_path(file_path, dpi=300)
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang="eng")
            text += f"\n--- Page {i+1} ---\n{page_text}"
    except Exception as e:
        st.warning(f"OCR failed for {file_path}: {e}")
    return text


def ocr_image(file_path):
    """Extract text from a single image using OCR."""
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        st.warning(f"OCR failed for {file_path}: {e}")
        return ""


def is_scanned_pdf(file_path):
    """Check if PDF has extractable text or is just scanned images."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        total_text = "".join([d.page_content.strip() for d in docs])
        return len(total_text) < 100  # Less than 100 chars → likely scanned
    except:
        return True


# -----------------------------
# 5. DOCUMENT INGESTION
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(file_paths, url):
    """Load documents (with OCR fallback), split, and create FAISS vector store."""
    documents = []

    for path in file_paths:
        ext = path.lower().split(".")[-1]

        # --- PDF ---
        if ext == "pdf":
            if is_scanned_pdf(path):
                st.info(f"🔍 Scanned PDF detected: {os.path.basename(path)} — running OCR...")
                ocr_text = ocr_pdf(path)
                documents.append(Document(
                    page_content=ocr_text,
                    metadata={"source": path, "type": "scanned_pdf"}
                ))
            else:
                loader = PyPDFLoader(path)
                documents.extend(loader.load())

        # --- DOCX ---
        elif ext == "docx":
            loader = Docx2txtLoader(path)
            documents.extend(loader.load())

        # --- TXT ---
        elif ext == "txt":
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

        # --- IMAGES ---
        elif ext in ["png", "jpg", "jpeg"]:
            st.info(f"🖼️ Image detected: {os.path.basename(path)} — running OCR...")
            ocr_text = ocr_image(path)
            documents.append(Document(
                page_content=ocr_text,
                metadata={"source": path, "type": "image"}
            ))

    # --- URL ---
    if url:
        try:
            url_loader = WebBaseLoader(url)
            documents.extend(url_loader.load())
        except Exception as e:
            st.sidebar.error(f"Could not load URL: {e}")

    if not documents:
        return None

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Add role-access metadata to each chunk
    for chunk in chunks:
        chunk.metadata["category"] = "General University"
        chunk.metadata["role_access"] = "Public"

    # Create embeddings + FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# Process uploaded files
if uploaded_files:
    os.makedirs("data/documents", exist_ok=True)
    saved_paths = []
    for uf in uploaded_files:
        path = os.path.join("data/documents", uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(path)

    with st.spinner("Indexing documents (may take time for scanned files)..."):
        st.session_state.vectorstore = build_vectorstore(saved_paths, url_input)

    if st.session_state.vectorstore:
        st.sidebar.success(f"✅ Indexed {len(saved_paths)} document(s)")

# -----------------------------
# 6. RAG CHAIN
# -----------------------------
SYSTEM_PROMPT = """You are IIUI Policy Assistant.

You answer questions using ONLY the authorized IIUI knowledge retrieved below.

Rules:
- Do NOT invent university policies.
- Do NOT use general knowledge to create an IIUI policy.
- If sufficient evidence is unavailable, clearly state that the information was not found.
- Prefer active and latest approved sources.
- Always provide the source (document title, section, page) when available.
- Respect the user's access level.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"]
)


def get_qa_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


# -----------------------------
# 7. MAIN CHAT UI
# -----------------------------
st.title("🎓 IIUI Policy Assistant")
st.caption(f"You are logged in as: **{user_role}**")
st.markdown("Ask questions about IIUI policies, academic regulations, admissions, fees and university procedures.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask your question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload at least one policy document or add a URL in the sidebar to begin.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching IIUI policies..."):
                try:
                    qa_chain = get_qa_chain(st.session_state.vectorstore)
                    result = qa_chain({"query": user_query})
                    answer = result["result"]
                    sources = result.get("source_documents", [])

                    # Display answer
                    st.markdown(answer)

                    # Display citations
                    if sources:
                        st.markdown("---")
                        st.markdown("**📖 Sources:**")
                        seen = set()
                        for doc in sources:
                            title = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "N/A")
                            key = f"{title}-{page}"
                            if key not in seen:
                                seen.add(key)
                                st.markdown(f"- `{os.path.basename(title)}` — Page {page}")

                    # Save full response to history
                    full_response = answer + "\n\n**Sources:** " + ", ".join(
                        [f"{os.path.basename(d.metadata.get('source','?'))} (p.{d.metadata.get('page','?')})"
                         for d in sources]
                    )
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Error: {e}")

# -----------------------------
# 8. FOOTER
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("MVP v1.0 — Grounded answers with citations. Built with Streamlit + LangChain + FAISS + OCR.")
