import ollama
import time
import json

# Updated target models optimized for edge-device constraints
target_models = [
    "llama3.2:1b",
    "qwen2.5:1.5b",      
    "gemma3:1b"       
]

# Base test prompts (No strict JSON enforcement, basic extraction prompts)
test_prompts = [
    "Extract the details from this text: Ticket #102: John Doe is requesting a refund of $45.50. His account email is j.doe99@gmail.com. Card ending in 4092 was charged incorrectly.",
    "Extract the details from this text: Refund request for order 881A. Charge was $120.00 to my Visa 1128. Please contact me at sarah.smith@company.org. - Sarah Smith.",
    "Extract the details from this text: I never received the item. Refund the 15.99 immediately. I don't want to give my email."
]

def run_inference_benchmark():
    all_metrics = []

    for model in target_models:
        print(f"\n--- Benchmarking: {model} ---")
        
        for i, prompt in enumerate(test_prompts):
            start_request = time.perf_counter()
            first_token_time = None
            
            try:
                # stream=True is required to capture TTFT accurately
                stream = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0},
                    stream=True
                )

                for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    
                    if chunk.get('done'):
                        metadata = chunk
                
                end_request = time.perf_counter()

                # Calculate Metrics
                ttft_ms = (first_token_time - start_request) * 1000
                gen_eval_sec = metadata.get('eval_duration', 1) / 1e9
                tokens_out = metadata.get('eval_count', 0)
                
                tps_gen = tokens_out / gen_eval_sec if gen_eval_sec > 0 else 0
                total_latency = end_request - start_request

                print(f"Prompt {i+1} | TTFT: {ttft_ms:.2f} ms | TPS: {tps_gen:.2f} | Latency: {total_latency:.2f} s")

                all_metrics.append({
                    "model": model,
                    "prompt_id": i+1,
                    "ttft_ms": round(ttft_ms, 2),
                    "tps_gen": round(tps_gen, 2),
                    "total_latency_s": round(total_latency, 2),
                    "tokens_generated": tokens_out
                })

            except Exception as e:
                print(f"Error benchmarking {model}: {str(e)}")

    # Dump results to JSON
    output_file = "phase1_inference_metrics.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"\n>> Phase 1 Complete. Metrics saved to {output_file}")

if __name__ == "__main__":
    run_inference_benchmark()