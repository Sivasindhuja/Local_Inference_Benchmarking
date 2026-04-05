import ollama
import time
import json
from pydantic import BaseModel, Field

# 1. THE SCHEMA
class TaskResponse(BaseModel):
    concept: str = Field(description="The subject of the task")
    category: str = Field(description="The technical domain (e.g. Security, DevOps)")
    difficulty_score: int = Field(ge=1, le=10)
    explanation: str = Field(description="The result of the task (explanation, extraction, or classification)")

# 2. THE MIXED-TASK DATASET (The "Standardized 30")
test_suite = [
    # --- EXPLANATIONS (Concepts) ---
    {"type": "Explanation", "prompt": "Explain Docker Volumes and why they are persistent."},
    {"type": "Explanation", "prompt": "How does a TCP/IP three-way handshake work?"},
    {"type": "Explanation", "prompt": "Explain the difference between Git Merge and Git Rebase."},
    {"type": "Explanation", "prompt": "What is the Virtual DOM in React?"},
    {"type": "Explanation", "prompt": "Explain the CAP theorem in distributed systems."},
    {"type": "Explanation", "prompt": "How does JWT (JSON Web Token) authentication stay secure?"},
    {"type": "Explanation", "prompt": "Explain the concept of 'Deadlocks' in Operating Systems."},
    {"type": "Explanation", "prompt": "What is Dependency Injection and why is it used?"},
    {"type": "Explanation", "prompt": "Explain the difference between REST and GraphQL."},
    {"type": "Explanation", "prompt": "What are WebSockets and when should you use them?"},

    # --- EXTRACTION (Data Recovery) ---
    {"type": "Extraction", "prompt": "Extract the email and server name from: 'Contact admin@devops.com on server PROD-DB-01'."},
    {"type": "Extraction", "prompt": "List all technologies mentioned: 'We use Python, PostgreSQL, and Redis in our stack.'"},
    {"type": "Extraction", "prompt": "Extract the version number and status from: 'Update 2.4.1 is currently STABLE.'"},
    {"type": "Extraction", "prompt": "Find the error code in: 'Critical failure: Process exited with code 0x8004 and status 500.'"},
    {"type": "Extraction", "prompt": "Extract the timestamp from: 'Log [2026-04-05 12:00:01] System started.'"},
    {"type": "Extraction", "prompt": "Identify the primary key in: 'Table Users(id: int, name: string, email: string)'"},
    {"type": "Extraction", "prompt": "Extract the hex color code from: 'The background should be #FF5733.'"},

    # --- LOGIC & CLASSIFICATION (Categorization) ---
    {"type": "Logic", "prompt": "Classify these as 'Frontend' or 'Backend': React, Django, Vue, Spring Boot, Node.js."},
    {"type": "Logic", "prompt": "Assign Priority (High/Low) to this bug: 'The login button turns blue on hover but works.'"},
    {"type": "Logic", "prompt": "Assign Priority (High/Low) to this bug: 'User passwords are visible in the URL bar.'"},
    {"type": "Logic", "prompt": "Classify these protocols: HTTP, SMTP, FTP, SSH, DNS."},
    {"type": "Logic", "prompt": "Identify the odd one out and why: Python, Java, HTML, C++."},
    {"type": "Logic", "prompt": "If A depends on B, and B depends on C, does A depend on C? Explain why."},
    {"type": "Logic", "prompt": "Classify the complexity (O(n), O(1), O(log n)) for: 'Finding an item in a sorted list via Binary Search.'"},

    # --- SYNTHESIS (Summarization) ---
    {"type": "Synthesis", "prompt": "Summarize in 10 words: 'CI/CD is a method to frequently deliver apps by introducing automation into the stages of app development.'"},
    {"type": "Synthesis", "prompt": "Convert this log to a summary: 'Worker A started. Task 1 failed. Retrying Task 1. Success.'"},
    {"type": "Synthesis", "prompt": "Describe the main goal of Agile development in one sentence."},
    {"type": "Synthesis", "prompt": "Summarize the benefit of Microservices over Monoliths in 15 words."},
    {"type": "Synthesis", "prompt": "Synthesize a status report: 'Database: UP, Server: UP, Cache: DOWN.'"},
    {"type": "Synthesis", "prompt": "Summarize why we use HTTPS instead of HTTP."}
]

results = []
# FEW-SHOT SYSTEM PROMPT (Required for 100% Success in Llama 3.2)
system_prompt = (
    "You are a technical assistant. Output ONLY JSON matching this schema: "
    "{'concept': 'name', 'category': 'domain', 'difficulty_score': 1-10, 'explanation': 'result'}. "
    "Example: {'concept': 'DNS', 'category': 'Networking', 'difficulty_score': 3, 'explanation': 'Translates domain names to IP addresses.'}"
)

print(f"--- Starting mistral Standardized 30-Prompt Benchmark ---")

for i, task in enumerate(test_suite):
    print(f"[{i+1}/30] Processing {task['type']}: {task['prompt'][:30]}...", end="", flush=True)
    
    start_time = time.perf_counter()
    try:
        response = ollama.chat(
            model="phi4-mini",
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': task['prompt']}
            ],
            format="json",
            options={'temperature': 0}
        )
        
        end_time = time.perf_counter()
        meta = response
        
        # Metrics
        tps = meta['eval_count'] / (meta['eval_duration'] / 1e9)
        ttft = meta['load_duration'] / 1e6 # Approximated load time in ms
        
        # Save with metadata for the report
        results.append({
            "type": task['type'],
            "prompt": task['prompt'],
            "status": "success",
            "tps": round(tps, 2),
            "latency": round(end_time - start_time, 2),
            "output": response['message']['content']
        })
        print(f" Done! ({tps:.2f} TPS)")

    except Exception as e:
        results.append({"type": task['type'], "prompt": task['prompt'], "status": "fail", "error": str(e)})
        print(f" FAILED.")

# Save the final data
with open("mistral_standardized_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n--- TEST COMPLETE: Data saved to mistral_standardized_results.json ---")