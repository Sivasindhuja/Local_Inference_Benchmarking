import ollama
import time
import pandas as pd
# We use Mistral 7B as the constant, changing only the quantization
# Make sure to run 'ollama pull' for each of these first
QUANT_MODELS = [
    "mistral:7b-instruct-v0.3-q4_0", 
    "mistral:7b-instruct-v0.3-q5_0"
]

test_prompt = "Explain the architecture of a PERN stack application in 100 words."

def quant_study():
    stats = []
    for model in QUANT_MODELS:
        print(f"Benchmarking {model}...")
        
        # Warm-up run (to load model into memory)
        ollama.chat(model=model, messages=[{'role': 'user', 'content': 'hi'}])
        
        start = time.perf_counter()
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': test_prompt}],
            options={'temperature': 0}
        )
        end = time.perf_counter()
        
        # Calculate Metrics
        gen_sec = response['eval_duration'] / 1e9
        tps = response['eval_count'] / gen_sec
        
        stats.append({
            "Quantization": model.split("-")[-1],
            "TPS (Speed)": round(tps, 2),
            "Total Latency": round(end - start, 2),
            "Character Count": len(response['message']['content']) # Proxy for quality/depth
        })

    df = pd.DataFrame(stats)
    print("\n--- QUANTIZATION TRADE-OFF DATA ---")
    print(df)
    df.to_csv("quantization_tradeoff.csv", index=False)

if __name__ == "__main__":
    quant_study()