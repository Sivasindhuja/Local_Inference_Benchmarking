#to calculate the time measurements(TTFT, TPS, Latency) and to implement the retry mechanism for deterministic JSON generation with validation.
import time
import ollama
from fastapi import FastAPI
import psutil
from pydantic import BaseModel, Field, ValidationError

app = FastAPI(title="Local AI Assistant: Engineering Benchmarks")

#schema for the structured response expected from the model, which will be used for validation in the retry mechanism using pydantic. The model is expected to return a JSON object with the specified fields, and if it fails to do so, the retry mechanism will provide feedback to the model to correct its output.
class TaskResponse(BaseModel):
    concept: str = Field(..., description="The subject of the task")
    category: str = Field(..., description="The technical domain")
    difficulty_score: int = Field(..., ge=1, le=10)
    explanation: str = Field(..., description="The technical explanation")

#default request schema for the benchmark endpoint, which includes the model to be used, the prompt for generation, and the temperature setting for controlling the randomness of the output. This schema will be used to validate incoming requests to ensure they contain the necessary information for processing.
class BenchmarkRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.0
    max_retries: int = 1

# --- HARDWARE MONITOR ---
def get_memory_usage():
    """Returns current system memory usage in GB."""
    return round(psutil.virtual_memory().used / (1024 ** 3), 2)


@app.post("/benchmark")
async def get_performance_metrics(req: BenchmarkRequest):
    start_request = time.perf_counter()
    first_token_time = None
    
    stream = ollama.chat(
        model=req.model,
        messages=[{'role': 'user', 'content': req.prompt}],
        options={'temperature': req.temperature},
        stream=True
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        if chunk.get('done'):
            metadata = chunk

    end_request = time.perf_counter()
    
    ttft_ms = (first_token_time - start_request) * 1000
    gen_sec = metadata.get('eval_duration') / 1e9
    tps = metadata.get('eval_count') / gen_sec if gen_sec > 0 else 0

    return {
        "metrics": {
            "ttft_ms": round(ttft_ms, 2),
            "tokens_per_second": round(tps, 2),
            "total_latency_sec": round(end_request - start_request, 2)
        },
        "model_info": metadata.get('model')
    }

@app.post("/test-stochasticity")
async def test_variance(model: str, concept: str):
    results = {}
    for temp in [0.0, 0.7]:
        explanations = []
        for _ in range(3): 
            resp = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': f"Explain {concept}"}],
                options={'temperature': temp}
            )
            explanations.append(resp['message']['content'])
        
        results[f"temp_{temp}"] = {
            "unique_count": len(set(explanations)),
            "samples": explanations
        }
    return results

@app.post("/generate")
async def generate_validated_content(req: BenchmarkRequest):
    system_prompt = (
        "Output ONLY JSON matching this schema: "
        "{'concept': str, 'category': str, 'difficulty_score': int, 'explanation': str}"
    )
    
    attempts = 0
    current_prompt = req.prompt
    start_time = time.perf_counter()
    mem_before = get_memory_usage()
    
    while attempts <= req.max_retries:
        response = ollama.chat(
            model=req.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': current_prompt}
            ],
            format="json",
            options={'temperature': 0}
        )
        
        content = response['message']['content']
        try:
            # Pydantic Validation
            validated = TaskResponse.model_validate_json(content)
            end_time = time.perf_counter()
            
            # Metrics Calculation
            dur_sec = response['eval_duration'] / 1e9
            tps = response['eval_count'] / dur_sec if dur_sec > 0 else 0
            
            return {
                "status": "success",
                "model": req.model,
                "retries": attempts,
                "metrics": {
                    "tps": round(tps, 2),
                    "latency_sec": round(end_time - start_time, 2),
                    "mem_usage_gb": get_memory_usage(),
                    "mem_delta_gb": round(get_memory_usage() - mem_before, 2)
                },
                "data": validated.model_dump()
            }
            
        except ValidationError as e:
            attempts += 1
            if attempts <= req.max_retries:
                current_prompt = f"INVALID JSON. Error: {str(e)}. Fix this JSON: {content}"
            else:
                return {"status": "failed", "error": "Max retries exceeded", "raw_output": content}


@app.get("/health")
async def health_check():
    """Requirement: Edge Deployment Connectivity Check"""
    return {"status": "online", "environment": "local_offline"}