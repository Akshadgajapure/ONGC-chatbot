# ONGC Knowledge Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that enables users to query ONGC technical documents, reports, and research papers using natural language. The system retrieves the most relevant document chunks from a FAISS vector database and generates accurate, context-aware responses using a Large Language Model.

---

## Features

- Document-based Question Answering
- Retrieval-Augmented Generation (RAG)
- Semantic Search using FAISS
- Multi-document PDF support
- Conversation History
- Source-aware responses
- Fast document retrieval
- Easy deployment using batch scripts

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Framework | Flask |
| LLM Framework | LangChain, LangGraph |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Vector Database | FAISS |
| LLM | Llama 3.2 |
| Document Processing | PyPDF2 / LangChain Document Loaders |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Package Manager | pip |

---

## Project Architecture

```

                PDFs
                  │
                  ▼
        Document Loader
                  │
                  ▼
        Text Chunking
                  │
                  ▼
     BGE Embedding Model
                  │
                  ▼
          FAISS Vector DB
                  │
                  ▼
User Question ──────────────► Similarity Search
                               │
                               ▼
                      Relevant Chunks
                               │
                               ▼
                        LangChain RAG
                               │
                               ▼
                         Llama 3.2
                               │
                               ▼
                      Final Response

```

---

## Folder Structure

```

ONGC-chatbot/
│
├── app.py
├── setup_db.py
├── requirements.txt
├── setup.bat
├── start.bat
├── index.html
├── ongc_logo.png
├── README.md
│
├── files/
│   ├── report1.pdf
│   ├── report2.pdf
│   └── ...
│
├── faiss_db/
│
└── chat_history.db
