# 🧠 MindEase — Mental Health RAG Chatbot
**CCAI 435 Deep Learning Project | University of Jeddah**

---

## 📌 Project Overview

MindEase is a **Retrieval-Augmented Generation (RAG)** chatbot that answers mental health questions.  
It retrieves the most relevant FAQ documents using semantic search (FAISS + Sentence Transformers),  
then generates a warm, grounded response using **Google Gemini 1.5 Flash**.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
[Embedding Model]  ← sentence-transformers (all-MiniLM-L6-v2)
    │
    ▼
[FAISS Vector Index]  ← cosine similarity search → Top-3 chunks
    │
    ▼
[Prompt Builder]  ← injects retrieved context + user query
    │
    ▼
[Gemini 1.5 Flash]  ← generates grounded, empathetic answer
    │
    ▼
[Streamlit UI]  ← displays answer + source excerpts
```

---

## 📂 Dataset

- **Name:** Mental Health FAQ for Chatbot  
- **Source:** [Kaggle — narendrageek/mental-health-faq-for-chatbot](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)  
- **Loaded via:** Public GitHub mirror (no manual download needed)  
- **Contents:** ~98 question-answer pairs covering anxiety, depression, therapy, PTSD, stress, and more

---

## 🚀 How to Run

### 1. Clone / place files
```
mental_health_rag/
├── app.py
├── requirements.txt
└── README.md
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
Streamlit will open automatically at: `http://localhost:8501`

---

## 🔑 API Key

The Google Gemini API key is already embedded in `app.py`.  
To change it, edit this line in `app.py`:
```python
GEMINI_API_KEY = "your-key-here"
```

---

## 🧪 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Retrieval Precision** | Top-K chunks retrieved are semantically relevant |
| **Answer Faithfulness** | Response stays grounded in retrieved context |
| **Response Quality** | Empathetic tone, clarity, and completeness |
| **Latency** | Average response time < 3 seconds |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI & Deployment | Streamlit |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (IndexFlatIP — cosine similarity) |
| Generative Model | Google Gemini 1.5 Flash |
| Dataset | Mental Health FAQ (Kaggle) |
| Language | Python 3.10+ |

---

## ⚠️ Disclaimer

MindEase is an educational prototype. It does **not** replace licensed mental health professionals.  
In case of crisis, please contact a qualified counselor or emergency services.
