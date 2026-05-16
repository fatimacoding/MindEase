# 🧠 MindEase — Mental Health RAG Chatbot

> A compassionate AI-powered mental health assistant built with Retrieval-Augmented Generation (RAG).  
> **CCAI 435 – Deep Learning · University of Jeddah**

---

## Overview

MindEase is a Streamlit-based chatbot that answers mental health questions by retrieving relevant passages from a curated FAQ knowledge base and generating warm, empathetic responses. It combines semantic search (FAISS + Sentence Transformers) with a large language model (DeepSeek) to ensure answers are grounded in evidence-based content rather than hallucinated.

---

## Features

- **RAG pipeline** — retrieves the top-K most relevant FAQ chunks before generating any response, keeping answers faithful to the source material
- **Semantic search** — `all-MiniLM-L6-v2` embeddings with a FAISS inner-product index for fast, accurate retrieval
- **Empathetic generation** — DeepSeek `deepseek-chat` prompted to respond warmly and flag when context is insufficient
- **Augmented knowledge base** — 15 supplemental FAQ entries covering topics underrepresented in the base dataset (PTSD, panic attacks, bipolar disorder, schizophrenia, sleep, mindfulness, meditation, and more)
- **Source transparency** — each bot reply shows the FAQ chunks it drew from
- **RAGAS evaluation** — automated pipeline measuring faithfulness, answer relevancy, context precision, and context recall

---

## Project Structure

```
.
├── app.py                  # Streamlit chat application
├── evaluate.py             # RAGAS evaluation pipeline
├── generated_answers.json  # Q/A/context triples from evaluation run
├── ragas_results.csv       # Per-question RAGAS metric scores
├── ragas_summary.txt       # Aggregated evaluation summary
├── README.md
└── requirements.txt
```

---

## Architecture

```
User query
    │
    ▼
Sentence Transformer (all-MiniLM-L6-v2)
    │  embed query
    ▼
FAISS Index (IndexFlatIP, cosine similarity)
    │  top-K chunks retrieved
    ▼
DeepSeek LLM (deepseek-chat)
    │  context-grounded generation
    ▼
MindEase response  +  source attribution
```

**Knowledge base:** [Mental Health FAQ (Kaggle)](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot) + 15 supplemental entries  
**Retriever:** FAISS `IndexFlatIP` with L2-normalised embeddings (top-5 chunks)  
**Generator:** DeepSeek `deepseek-chat` via OpenAI-compatible API

---

## Evaluation Results (RAGAS v2)

Evaluated on 15 test questions using DeepSeek as the judge LLM.

| Metric | Score |
|---|---|
| Faithfulness | **0.9798** |
| Answer Relevancy | **0.9445** |
| Context Precision | **0.9333** |
| Context Recall | **0.9822** |

All four metrics exceed 0.93, indicating the system produces highly faithful, relevant, and well-grounded answers.

---

## Getting Started

### Prerequisites

- Python 3.9+
- A DeepSeek API key

### Installation

```bash
pip install streamlit pandas numpy requests openai \
            sentence-transformers faiss-cpu
```

### Run the app

```bash
streamlit run app.py
```

The app fetches the FAQ dataset and builds the FAISS index on first launch (cached for subsequent runs).

---

## Running the Evaluation

Install additional dependencies:

```bash
pip install ragas datasets langchain-openai langchain-community
```

Then run:

```bash
python evaluate.py
```

This produces three output files: `generated_answers.json`, `ragas_results.csv`, and `ragas_summary.txt`.

---

## Configuration

Key constants in both `app.py` and `evaluate.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `TOP_K` | `5` (eval) / `3` (app) | Retrieved chunks per query |
| `DEEPSEEK_MODEL` | `deepseek-chat` | LLM for generation and judging |

> **Note:** The API key is currently hardcoded. For production use, load it from an environment variable or secrets manager (e.g. `st.secrets` in Streamlit).

---

## Disclaimer

MindEase is an educational prototype and **does not replace professional mental health care**. If you or someone you know is in crisis, please contact a licensed mental health professional or a crisis helpline immediately.

---

## Dataset

[Mental Health FAQ for Chatbot — Kaggle](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)

---

## Tech Stack

| Component | Library / Service |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers` |
| Vector search | FAISS (`faiss-cpu`) |
| LLM | DeepSeek (`deepseek-chat`) |
| Evaluation | RAGAS |
| LLM integration | LangChain + `langchain-openai` |
