# PII Extraction from Refund Tickets

An offline API system that extracts structured Personally Identifiable Information (PII) and refund details from unstructured customer support tickets using local Small Language Models (SLMs).

The system converts messy refund messages into clean JSON containing:

* Customer Name
* Email Address
* Refund Amount
* Currency

Built to run locally for better privacy, lower cost, and reduced latency.

---

## Documentation

 Full Engineering Report (PDF): https://github.com/Sivasindhuja/Local_Inference_Benchmarking/blob/main/docs/Technical_Documentation.pdf

---

## Why This Project?

Customer support refund tickets often contain important information hidden inside free-text messages, forwarded emails, signatures, and noisy content.

Manual review is slow and error-prone.

This project automates extraction of refund-related fields from raw tickets using local language models with validation and fallback routing.

---

## Key Features

* Offline local inference using Ollama Ollama
* REST API built with FastAPI FastAPI
* Structured JSON output
* Schema validation using Pydantic Pydantic
* Retry logic for invalid JSON responses
* Fallback routing for difficult edge cases
* Benchmark-driven model selection
* Privacy-friendly local processing

---

## Models Evaluated

* Meta Llama 3.2 (1B / 3B)
* Alibaba Cloud Qwen 2.5 (1.5B)
* Google Gemma 3 (1B)
* DeepSeek DeepSeek-R1

Final primary model selected: **Qwen 2.5 (1.5B)**

---

## Accuracy Improvement Journey

| Stage                  | Accuracy   |
| ---------------------- | ---------- |
| Baseline Qwen 2.5      | 58.30%     |
| Currency Prompt Fix    | 70.80%     |
| Few-shot Prompting     | 79.17%     |
| API Cascade + Fallback | **83.30%** |

---

## Architecture

```text id="ybk3kv"
Incoming Refund Ticket
        ↓
Primary Model (Qwen 2.5)
        ↓
Schema Validation
        ↓
If Invalid → Retry
If Complex / Low Confidence → Fallback Model (Llama 3.2 3B)
        ↓
Final Structured JSON Output
```

---

## Example Input

```text id="2hlbaj"
Hi team, this is Priya Sharma.
My email is priya.sharma@gmail.com.
Please refund $24.99 for duplicate charge.
```

## Example Output

```json id="it7v8k"
{
  "name": "Priya Sharma",
  "email": "priya.sharma@gmail.com",
  "refund_amount": 24.99,
  "currency": "USD"
}
```

---

## How to Run This Project

### 1. Clone the Repository

```bash id="5q5c13"
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install Python Dependencies

```bash id="vxw5q1"
pip install -r requirements.txt
```

### 3. Install and Start Ollama

Download Ollama, then pull required models:

```bash id="86o1lc"
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
```

### 4. Start the API Server

```bash id="fd0u0y"
uvicorn app:app --reload
```

### 5. Test the API

```bash id="3fthq2"
python test_api.py
```

---

## Benchmark Dataset

Custom 24-case evaluation suite covering:

* Missing fields
* Multiple conflicting values
* Forwarded email chains
* Quoted replies
* Signature noise
* Negation / denied refunds
* OCR corrupted text
* Multilingual text
* Currency symbols
* Fake embedded JSON

---

## Tech Stack

* Python
* FastAPI FastAPI
* Pydantic Pydantic
* Ollama Ollama
* Local LLMs
* JSON APIs

---

## Remaining Challenges

* Obfuscated emails (`name at gmail dot com`)
* Deep nested email threads
* Multiple customers in one ticket
* False extraction from visible tokens
* Ambiguous business context

---

## Future Improvements

* Human review queue
* Regex + LLM hybrid extraction
* Confidence scoring
* Fine-tuned domain model
* PostgreSQL Global Development Group PostgreSQL logging & monitoring

---

## What This Project Demonstrates

* Practical LLM system design
* Prompt engineering
* Multi-model routing
* Validation pipelines
* Offline AI deployment
* Benchmark-driven engineering

---

## Author

Built as an applied AI engineering project focused on real-world document understanding and local model deployment.
