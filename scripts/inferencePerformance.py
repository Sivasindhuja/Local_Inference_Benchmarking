import ollama
import time

def measure_professional_performance(model_name, prompt):
    print(f"\n--- Professional Benchmark: {model_name} ---")
    
    start_request = time.perf_counter()
    first_token_time = None
    full_response = ""

    # Using stream=True to capture the moment the model starts talking
    stream = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0},
        stream=True
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        
        full_response += chunk['message']['content']
        
        # The final chunk contains the metadata
        if chunk.get('done'):
            metadata = chunk

    end_request = time.perf_counter()

    # Calculation logic
    ttft = (first_token_time - start_request) * 1000 # convert to ms
    
    # Metadata parsing (Ollama returns ns, we convert to s)
    prompt_eval_sec = metadata.get('prompt_eval_duration') / 1e9
    gen_eval_sec = metadata.get('eval_duration') / 1e9
    tokens_out = metadata.get('eval_count')
    tokens_in = metadata.get('prompt_eval_count')

    # Metrics
    tps_gen = tokens_out / gen_eval_sec if gen_eval_sec > 0 else 0
    tps_prompt = tokens_in / prompt_eval_sec if prompt_eval_sec > 0 else 0

    print(f"Time to First Token (TTFT): {ttft:.2f} ms")
    print(f"Prompt Processing Speed: {tps_prompt:.2f} tokens/sec")
    print(f"Generation Speed (TPS): {tps_gen:.2f} tokens/sec")
    print(f"Total Latency: {end_request - start_request:.2f} s")

    return {
        "model": model_name,
        "ttft_ms": ttft,
        "tps": tps_gen,
        "tokens": tokens_out
    }

# Run the test
prompt = "Explain CI/CD in 50 words."
measure_professional_performance("mistral", prompt)