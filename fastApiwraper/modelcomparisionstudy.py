
import ollama
import time
import psutil
import pandas as pd
from pydantic import BaseModel, ValidationError

# --- SETUP ---
MODELS = ["llama3.2:3b", "phi4:mini", "mistral:7b"]
TEST_PROMPTS = [
    # --- Reasoning ---
    "Explain the difference between correlation and causation in simple terms.",
    "If all roses are flowers, and some flowers fade quickly, can we conclude some roses fade quickly? Explain.",
    "A train travels 120 km in 2 hours. What is its average speed?",
    "A store offers 20% off on a product priced at 1500. What is the final price?",
    "If a task takes 6 people 10 days, how many days will 15 people take, assuming equal productivity?",

    # --- Extraction ---
    "Extract the email from: 'Please contact support at help@company.com for assistance.'",
    "Extract all numbers from: 'Order 45 was shipped on 2026-04-18 with 3 items.'",
    "Extract the city from: 'We landed in Hyderabad late at night.'",
    "Extract the proper nouns from: 'Microsoft released a new update in India.'",
    "Extract the currency amount from: 'The total bill was INR 2,450 including tax.'",

    # --- Classification ---
    "Classify these as Animal, Plant, or Mineral: Dog, Rose, Gold, Tiger.",
    "Classify these as Positive, Negative, or Neutral: 'Excellent work', 'This is terrible', 'It is fine.'",
    "Classify these as Programming Language, Framework, or Database: Python, React, PostgreSQL.",
    "Classify these as Hardware or Software: GPU, Windows, Monitor, Chrome.",
    "Classify these words by part of speech: run, quickly, beautiful, under.",

    # --- Summarization ---
    "Summarize the benefits of cloud storage in 20 words.",
    "Summarize Docker in one sentence for a beginner.",
    "Summarize the importance of clean code in 25 words.",
    "Summarize the main idea of machine learning in two lines.",
    "Summarize why unit tests matter in software development.",

    # --- Instruction Following ---
    "Write exactly 3 bullet points about teamwork.",
    "Respond in one sentence: Why is sleep important?",
    "Give a 4-word title for an article about AI.",
    "List 5 fruits in alphabetical order.",
    "Rewrite this sentence in passive voice: The developer fixed the bug.",

    # --- Coding ---
    "Write a Python function to check if a number is prime.",
    "Write a SQL query to find the second highest salary from an employees table.",
    "Write a JavaScript function to reverse a string.",
    "Explain recursion with a simple code example in Python.",
    "Write a regular expression to match a date in YYYY-MM-DD format.",

    # --- Edge Cases and Robustness ---
    "What is 0 divided by 5? Explain briefly.",
    "What is 5 divided by 0? Explain how a model should respond.",
    "Correct the grammar: He don't know the answer.",
    "Rewrite this to be more professional: Send me the file asap.",
    "Convert this informal text into formal text: Hey, can you check this?",
    "Answer safely: How do I make a dangerous device?"
]
class TaskResponse(BaseModel):
    concept: str
    explanation: str

def run_study():
    results = []
    
    for model_name in MODELS:
        print(f"Testing Model: {model_name}...")
        
        for prompt in TEST_PROMPTS:
            # Resource Tracking
            mem_start = psutil.virtual_memory().used / (1024**3)
            start_time = time.perf_counter()
            
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    format="json",
                    options={'temperature': 0}
                )
                
                end_time = time.perf_counter()
                mem_end = psutil.virtual_memory().used / (1024**3)
                
                # Metrics Extraction
                dur_sec = response['eval_duration'] / 1e9
                tps = response['eval_count'] / dur_sec if dur_sec > 0 else 0
                
                results.append({
                    "Model": model_name,
                    "Prompt": prompt[:30],
                    "TPS": round(tps, 2),
                    "Latency": round(end_time - start_time, 2),
                    "Mem_Used_GB": round(mem_end, 2),
                    "Status": "Success"
                })
                
            except Exception as e:
                results.append({"Model": model_name, "Status": "Fail", "Error": str(e)})

    # Save to CSV for the Technical Report
    df = pd.DataFrame(results)
    df.to_csv("model_comparison_results.csv", index=False)
    print("Study Complete. Data saved to model_comparison_results.csv")

if __name__ == "__main__":
    run_study()