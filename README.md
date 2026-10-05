# 🎓 UniMind | Intelligent Academic Knowledge Assistant

UniMind is an offline-capable, deterministic, citation-backed Retrieval-Augmented Generation (RAG) system engineered for academic institutions, universities, and compliance-driven organizations. It allows students and faculty to query complex regulations, syllabi, exam ordinances, and policies with zero hallucinations.


Live Link : https://github.com/sampreethg/-Intelligent-University-Knowledge-Assistant
---

## 🚀 Key Features

- **100% Offline & Local Execution:** Powered by `qwen2.5-coder:7b` via Ollama and `all-MiniLM-L6-v2` embeddings—no third-party cloud LLM API keys or data leakage.
- **Strict Grounded Answering:** Enforces temperature `0.0` and inline citations (`[Document, Page X]`). If information is absent from indexed files, the system states it cannot find it rather than guessing.
- **Dual-Store Architecture:** Synchronized FAISS `IndexIDMap2` vector index mapped 1:1 with an SQLite metadata repository.
- **Enterprise Dark UI/UX:** Built with Streamlit, custom CSS glassmorphism, response telemetry tracking, confidence score percentage badges, and an interactive document chunk inspector.
- **Live Remote Demonstration:** Built-in Cloudflare Tunnel support for zero-cost, instant HTTPS sharing.

---

## 🏗️ Architecture

```text
User Query / Document Ingestion
          │
          ▼
 [ Streamlit UI Portal ]
   │                │
   ▼ (Upload)       ▼ (Query)
 [ Ingestion & ]    [ Sentence-Transformers ]
 [ Sliding Chunk]        (all-MiniLM-L6-v2)
   │                        │
   ▼                        ▼
 [ FAISS IndexIDMap2 ] ◄──► [ SQLite (documents.db) ]
   │ (Top-k Chunks + Scores)
   ▼
 [ Grounded Prompting Engine ]
   │
   ▼
 [ Ollama (qwen2.5-coder:7b) ]
   │ (Streaming Tokens)
   ▼
 [ Verified Answer + Telemetry + Badges ]
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Frontend** | Streamlit | Responsive dashboard with custom CSS, floating popover guide, modals |
| **Embeddings** | Sentence-Transformers | `all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Vector Index** | FAISS | `IndexIDMap2` with `IndexFlatIP` (Cosine Similarity) |
| **Relational Metadata** | SQLite | Tracks uploaded files, chunks, page mappings, and foreign keys |
| **LLM Inference** | Ollama | `qwen2.5-coder:7b` running locally on CPU |
| **Document Parsing** | PyMuPDF, python-docx | Parses `.pdf`, `.docx`, `.txt` preserving page numbers |
| **Tunnel / Hosting** | Cloudflare Tunnel | Public HTTPS endpoint for zero-cost remote evaluation |

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.10+ installed
- [Ollama](https://ollama.com/) installed with `qwen2.5-coder:7b`:
  ```bash
  ollama pull qwen2.5-coder:7b
  ```

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/sampreethg/-Intelligent-University-Knowledge-Assistant.git
cd -Intelligent-University-Knowledge-Assistant

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Streamlit Configuration
Ensure `.streamlit/config.toml` exists to permit WebSocket traffic:
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
port = 8501
```

### 4. Running the Application
Open two terminal windows:

**Terminal 1 (Start Ollama):**
```powershell
ollama serve
```

**Terminal 2 (Start Streamlit):**
```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

### 5. Exposing for Remote Demo (Cloudflare Tunnel)
```powershell
.\cloudflared.exe tunnel --url http://localhost:8501
```
Use the generated `https://*.trycloudflare.com` URL to test on mobile or share with reviewers.

---

## 🔒 Error Handling & Safety
- **Missing / Crashed Ollama Engine:** Handled gracefully via user-friendly UI warning banners.
- **Corrupted or Scanned Documents:** Catches unparseable PDFs, empty buffers, and password-protected files.
- **Index Synchronization:** Document deletion removes records from both SQLite and FAISS simultaneously.
