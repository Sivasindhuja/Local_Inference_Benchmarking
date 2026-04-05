import ollama
import json
import time
from pydantic import BaseModel, Field, ValidationError

class TaskResponse(BaseModel):
    concept: str
    category: str
    difficulty_score: int
    explanation: str

def run_eval_batch(model_name, prompts, system_prompt):
    results = {"success": 0, "fail": 0, "errors": []}
    
    for concept in prompts:
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Explain {concept}"}
                ],
                format="json",
                options={'temperature': 0}
            )
            TaskResponse.model_validate_json(response['message']['content'])
            results["success"] += 1
        except Exception as e:
            results["fail"] += 1
            results["errors"].append({"concept": concept, "error": str(e)})
            
    return results

# --- DATASET ---
test_prompts = [
    "Docker Volumes", "JWT", "SQL Joins", "Recursion", "CI/CD", 
    "Load Balancing", "REST APIs", "Git Rebase", "Deadlocks", "Binary Search"
] # Start with 10 for a quick test

# --- ITERATION 1: Basic Prompt ---
sys_v1 = "You are a technical assistant. Output JSON."

# --- ITERATION 2: Prompt with Examples (Few-Shot) ---
sys_v2 = (
    "You are a technical assistant. Output ONLY JSON. "
    "Required keys: concept (str), category (str), difficulty_score (int), explanation (str). "
    "Example: {'concept': 'RAM', 'category': 'Hardware', 'difficulty_score': 2, 'explanation': 'Short term memory.'}"
)

print("Running Iteration 1...")
report_v1 = run_eval_batch("mistral", test_prompts, sys_v1)
print(f"v1 Success Rate: {report_v1['success'] / len(test_prompts) * 100}%")

print("\nRunning Iteration 2...")
report_v2 = run_eval_batch("mistral", test_prompts, sys_v2)
print(f"v2 Success Rate: {report_v2['success'] / len(test_prompts) * 100}%")

#11:12

