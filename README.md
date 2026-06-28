# 🎓 Adaptive RAG Academic Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** system for answering questions from academic PDF documents using semantic search, adaptive context optimization, and a locally hosted Large Language Model (LLM).

Unlike traditional document chatbots, this project introduces a **Binary Search-Based Context Optimizer (BSCO)** that minimizes the amount of context passed to the LLM while preserving answer quality. The system retrieves only the most relevant document chunks, generates grounded responses, provides page-level citations, and estimates confidence for every answer.

---

## 🌟 Overview

Adaptive RAG Academic Chatbot is designed to help students, researchers, and educators interact with academic documents efficiently. Users can upload PDF documents, build a searchable vector database, and ask natural language questions. The chatbot answers exclusively from the uploaded document, reducing hallucinations and ensuring document-grounded responses.

The entire pipeline runs locally using **Ollama**, making it privacy-friendly without requiring cloud APIs.

---

# ✨ Features

- 📄 Upload and process academic PDF documents
- 🔍 Semantic search using FAISS vector similarity
- 🧩 Adaptive document chunking for improved retrieval
- ⚡ Binary Search-Based Context Optimizer (BSCO)
- 🧠 Local LLM inference using Ollama (Llama 3.2)
- 📚 Automatic citation generation with supporting page numbers
- 📊 Confidence estimation for every generated answer
- 💬 Interactive Streamlit web interface
- 🔒 Fully local execution (no cloud dependency)
- ⚙️ Modular architecture for easy extension and research

---

# 🚀 Key Innovations

Unlike conventional RAG systems, this project introduces several research-oriented improvements:

### Binary Search-Based Context Optimizer (BSCO)

Instead of sending every retrieved chunk to the LLM, BSCO dynamically finds the **smallest sufficient context** required to answer the user's question. This approach:

- Reduces prompt size
- Improves response quality
- Lowers inference cost
- Speeds up generation
- Reduces irrelevant context

### Adaptive Chunking

Rather than using fixed-size chunks, documents are segmented while preserving meaningful academic structures such as paragraphs and sections, resulting in more accurate retrieval.

### Citation Generation

Every answer includes supporting page references, making responses verifiable and trustworthy.

### Confidence Estimation

A confidence score is computed based on retrieval similarity, helping users judge the reliability of generated answers.

---

# 🏗️ System Architecture

```text
                 Academic PDF
                      │
                      ▼
              PDF Processing
                      │
                      ▼
            Adaptive Chunking
                      │
                      ▼
          Embedding Generation
        (BAAI/bge-small-en-v1.5)
                      │
                      ▼
           FAISS Vector Database
                      │
                      ▼
             Semantic Retrieval
                      │
                      ▼
 Binary Search-Based Context Optimizer
          (BSCO Context Reduction)
                      │
                      ▼
             Prompt Construction
                      │
                      ▼
          Ollama (Llama 3.2 : 3B)
                      │
                      ▼
            Answer Generation
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 Citation Generation      Confidence Estimation
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Streamlit User Interface
```

---

# 📂 Project Structure

```text
Adaptive-RAG-Academic-Chatbot/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── pdf_processor.py
│   ├── adaptive_chunker.py
│   ├── embedding.py
│   ├── vector_database.py
│   ├── semantic_search.py
│   ├── binary_search.py
│   ├── prompt_builder.py
│   ├── answer_generator.py
│   ├── citations.py
│   ├── confidence.py
│   └── pipeline.py
│
├── uploads/
├── vector_db/
├── tests/
└── assets/
```

---

# 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Embedding Framework | Sentence Transformers |
| Vector Database | FAISS |
| Large Language Model | Llama 3.2 (3B) |
| LLM Runtime | Ollama |
| Numerical Computing | NumPy |
| Deep Learning | PyTorch |

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/raghavendra1500/Adaptive-RAG-Academic-Chatbot.git

cd Adaptive-RAG-Academic-Chatbot
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.\.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download Ollama from:

https://ollama.com/download

---

## Download the LLM

```bash
ollama pull llama3.2:3b
```

---

## Start Ollama

```bash
ollama serve
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser and navigate to:

```
http://localhost:8501
```

---

# 💡 Usage

1. Launch the application.
2. Upload an academic PDF.
3. Build the vector database.
4. Ask questions in natural language.
5. Receive:
   - AI-generated answer
   - Supporting page citations
   - Confidence score
   - Retrieved context

---

# 📌 Example

### Question

```
What are the Program Outcomes?
```

### Response

```
Answer:
The uploaded academic document does not explicitly define Program Outcomes.

Supporting Pages:
Page 67, Page 102

Confidence:
Medium
```

---

# 📈 Research Contributions

This project extends a conventional RAG pipeline by introducing:

- Adaptive Chunking
- Binary Search-Based Context Optimizer (BSCO)
- Semantic Retrieval using FAISS
- Citation Generation
- Confidence Estimation
- Modular RAG Architecture
- Local LLM Inference using Ollama

These components collectively improve retrieval efficiency, reduce unnecessary context, and produce more transparent, document-grounded responses.

---

# 🔮 Future Work

- Hybrid Retrieval (Dense + Sparse Search)
- Cross-Encoder Reranking
- Multi-document Question Answering
- OCR Support for Scanned PDFs
- PDF Page Highlighting
- Conversation Memory
- Chat History
- User Authentication
- Cloud Deployment
- Multi-modal Document Support
- Fine-tuned Academic Language Models
- Evaluation Dashboard for Retrieval Metrics

---

# 📷 Screenshots

Add screenshots in the `assets/` folder.

Suggested images:

```
assets/home.png

assets/upload.png

assets/chat.png

assets/results.png

assets/confidence.png
```

Or include a short demo GIF:

```
assets/demo.gif
```

---

# 📜 License

This project is licensed under the MIT License.

See the LICENSE file for more details.

---

# 👨‍💻 Authors

**K Sai Raghavendra**

**B Siddartha**

**C Nihal Reddy**

---

# 🙏 Acknowledgements

This project makes use of several outstanding open-source technologies:

- Ollama
- Meta Llama
- Hugging Face
- Sentence Transformers
- FAISS
- PyMuPDF
- Streamlit
- PyTorch
- NumPy

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!

Your support helps improve and grow the project.
