

---

# SLM-Benchmarking v1.0: Local Inference & Reliability Framework

**Live Dashboard:** [https://sivasindhuja.github.io/Local_Inference_Benchmarking/]

This repository contains a comprehensive benchmarking suite and research study on the practicalities of running **Small Language Models (SLMs)** on consumer-grade integrated graphics. Moving beyond simple chat interfaces, this project implements a **Reliability Layer** using Pydantic to ensure local models are production-ready for automated software pipelines.

---

##  System Architecture
* **Hardware:** Intel(R) Iris(R) Xe Graphics (iGPU) | 16GB LPDDR4x RAM (8GB Shared VRAM).
* **Inference Engine:** [Ollama](https://ollama.com/) v0.5+.
* **Validation Layer:** Pydantic v2.x.
* **Language:** Python 3.11.

---

##  Key Features

### 1. Multi-Model Performance Analytics
We benchmarked **Llama 3.2 (3B)**, **Phi-4 Mini (3.8B)**, and **Mistral 7B (v0.3)** across three critical KPIs:
* **Tokens Per Second (TPS):** Raw generation throughput.
* **Time to First Token (TTFT):** Initial response latency (snappiness).
* **Total Response Latency:** End-to-end execution time for complex tasks.

### 2. The Pydantic Reliability Loop
Small models often struggle with instruction following. We implemented a **Defensive Programming** wrapper that:
* Enforces a strict JSON schema.
* Validates outputs using Pydantic models.
* **Self-Correction:** Automatically re-prompts the model once with specific error feedback if a schema is violated.
* Fails gracefully with standardized null-objects to prevent system crashes.

### 3. Stochasticity Control
A study on how **Temperature ($T$)** affects reliability. We documented why $T=0$ (Greedy Decoding) is mandatory for structured data tasks, as $T=0.7$ led to a significant increase in schema hallucinations.

---

##  Benchmarking Results (Warm Start)

| Model | TPS | TTFT | JSON Adherence (Few-Shot) |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 (3B)** | **13.52** | 284.35 ms | **100%** |
| **Phi-4 Mini (3.8B)** | 7.42 | 255.10 ms | **100%** |
| **Mistral 7B (v0.3)** | 7.71 | **204.29 ms** | 86% |

---

##  Engineering Challenges & Lessons Learned

### **The "Cold Start" Penalty**
Phi-4 Mini exhibited a **62-second load time** on the Intel iGPU. We solved this by implementing a **Warm-up Pulse** to ensure model weights are resident in Shared VRAM before benchmarking begins.

### **Zero-Shot Incompetence**
Llama 3.2 scored **0% success** on JSON tasks zero-shot. We developed a **Few-Shot System Prompting** strategy that provided a structural anchor, resulting in a jump to **100% success**.

### **Memory Fragmentation (Status 500)**
Running 7B models on 8GB Shared VRAM frequently caused runner termination. We implemented a **"Clean Slate" Protocol** to manage background RAM usage and deallocate GPU memory between runs.

---

## Repository Structure
* `inferencePerformance.py`: Measures TPS, TTFT, and Latency.
* `structured_output.py`: The Pydantic validation and retry logic implementation.
* `test_prompts.py`: The standardized 30-prompt evaluation suite.
* `temperature.py`: Script for measuring output variance.
* `index.html`: The source code for the [Live Dashboard].

---

##  How to Run
1.  Install [Ollama](https://ollama.com/).
2.  Clone this repo.
3.  Install dependencies: `pip install pydantic ollama`.
4.  Run the benchmark: `python test_prompts.py`.

---

**Author:** Siva Sindhuja Tsundupalli  

---
