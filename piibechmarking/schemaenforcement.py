import ollama
import time
import json
from pydantic import BaseModel, Field

# 1. NEW SCHEMA: PII Data Sanitization
class ExtractedPII(BaseModel):
    customer_name: str | None =  Field(description="Full name of the customer. Return 'None' if missing.")
    email_address: str | None =  Field(description="Email address. Return 'None' if missing.")
    credit_card_last_four: str | None =  Field(description="4-digit card ending. Return 'None' if missing.")
    claim_amount: float = Field(description="The numeric amount requested for refund.")

# 2. DOMAIN-SPECIFIC DATASET (Refund Claims)
# Expand this to 30 items for your final report
test_suite = [
    {
        "prompt": "Ticket #102: John Doe is requesting a refund of $45.50. His account email is j.doe99@gmail.com. Card ending in 4092 was charged incorrectly.",
        "ground_truth": {"customer_name": "John Doe", "email_address": "j.doe99@gmail.com", "credit_card_last_four": "4092", "claim_amount": 45.50}
    },
    {
        "prompt": "Refund request for order 881A. Charge was $120.00 to my Visa 1128. Please contact me at sarah.smith@company.org. - Sarah Smith.",
        "ground_truth": {"customer_name": "Sarah Smith", "email_address": "sarah.smith@company.org", "credit_card_last_four": "1128", "claim_amount": 120.00}
    },
    {
        "prompt": "I never received the item. Refund the 15.99 immediately. I don't want to give my email.",
        "ground_truth": {"customer_name": "None", "email_address": "None", "credit_card_last_four": "None", "claim_amount": 15.99}
    }
]

system_prompt = (
    "You are an offline data sanitization agent. Extract PII from the text. "
    "Output ONLY valid JSON matching this schema: "
    "{'customer_name': str, 'email_address': str, 'credit_card_last_four': str, 'claim_amount': float}. "
    "If a field is missing, output 'None' for strings or 0.0 for floats."
)

# Target SLMs for your 16GB i5 Hardware
target_models = ["llama3.2:1b", "qwen2.5:1.5b", "deepseek-r1:1.5b", "smollm2:1.7b"]

def run_sanitization_benchmark():
    all_results = {}

    for model in target_models:
        print(f"\n--- Benchmarking: {model} ---")
        model_results = []
        success_count = 0

        for i, task in enumerate(test_suite):
            start_time = time.perf_counter()
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': task['prompt']}
                    ],
                    format="json",
                    options={'temperature': 0}
                )
                end_time = time.perf_counter()
                
                # Validation & Metrics
                raw_json = json.loads(response['message']['content'])
                validated_data = ExtractedPII(**raw_json)
                
                meta = response
                tps = meta.get('eval_count', 0) / (meta.get('eval_duration', 1) / 1e9)
                
                # Basic Accuracy Check
                is_accurate = (
                    validated_data.email_address == task['ground_truth']['email_address'] and 
                    validated_data.claim_amount == task['ground_truth']['claim_amount']
                )

                if is_accurate: success_count += 1

                model_results.append({
                    "prompt_id": i+1,
                    "tps": round(tps, 2),
                    "latency_s": round(end_time - start_time, 2),
                    "schema_valid": True,
                    "data_accurate": is_accurate,
                    "extracted": validated_data.model_dump()
                })

            except Exception as e:
                model_results.append({"prompt_id": i+1, "schema_valid": False, "error": str(e)})

        accuracy_rate = (success_count / len(test_suite)) * 100
        print(f">> Schema Valid & Data Accurate: {accuracy_rate}%")
        all_results[model] = model_results

    with open("slm_sanitization_report.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    run_sanitization_benchmark()