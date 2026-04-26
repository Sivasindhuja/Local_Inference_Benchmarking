import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, EmailStr, ValidationError, Field
import os
from fastapi import FastAPI, HTTPException
try:
    import ollama
except ImportError:
    raise SystemExit("Please install ollama Python package: pip install ollama")

CONFIG_FILE = "prompts.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Configuration file {CONFIG_FILE} is missing.")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    PROMPT_CONFIG = json.load(f)

ACTIVE_VERSION = PROMPT_CONFIG.get("active_version", "v1.2")
PROMPTS = PROMPT_CONFIG.get("versions", {})

if not PROMPTS or ACTIVE_VERSION not in PROMPTS:     
    raise ValueError(f"No prompts found for active version: {ACTIVE_VERSION}")
#  Application Setup 
app = FastAPI(title="PII Extraction Engine for refund requests", version="1.2.0")

PRIMARY_MODEL = "qwen2.5:1.5b"
FALLBACK_MODEL = "llama3.2:3b"

#input schema
class TicketRequest(BaseModel):
    ticket_text: str = Field(..., description="The raw text of the customer support ticket.")
    prompt_version: Optional[str] = Field(default=ACTIVE_VERSION, description="Target prompt version.")
#output schema
class RefundExtraction(BaseModel):
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None) 
    refund_amount: Optional[float] = Field(default=None)
    currency: Optional[str] = Field(default=None)


    #normalisation
CURRENCY_MAP = {
    "$": "USD",
    "₹": "INR",
    "€": "EUR",
    "£": "GBP",
}


def normalize_currency(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip().upper()
    if v in {"USD", "INR", "EUR", "GBP"}:
        return v
    return CURRENCY_MAP.get(v, v)


def normalize_name(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = re.sub(r"\s+", " ", v).strip()
    return v or None


def normalize_email(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    return v.strip().lower()


def normalize_amount(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(Decimal(str(v)).quantize(Decimal("0.01")))
    except (InvalidOperation, ValueError):
        return None


def normalize_record(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": normalize_name(d.get("name")),
        "email": normalize_email(d.get("email")),
        "refund_amount": normalize_amount(d.get("refund_amount")),
        "currency": normalize_currency(d.get("currency")),
    }

def extract_json_candidate(text: str) -> Optional[str]:
    text = text.strip()
    fenced = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fenced: 
        return fenced[0]
    start, end = text.find("{"), text.rfind("}")
    return text[start:end+1] if start != -1 and end != -1 and end > start else None

def parse_and_validate(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(raw)
    except Exception:
        candidate = extract_json_candidate(raw)
        if not candidate: return None, "Invalid JSON structure."
        try:
            obj = json.loads(candidate)
        except Exception:
            return None, "Invalid JSON structure inside markdown."
            
    try:
        parsed = RefundExtraction.model_validate(obj)
        return {
            "name": normalize_name(parsed.name),
            "email": normalize_email(parsed.email),
            "refund_amount": normalize_amount(parsed.refund_amount),
            "currency": normalize_currency(parsed.currency),
        }, None
    except ValidationError as e:
        return None, f"Pydantic Schema Error: {str(e)}"

# Core Inference with Error Reprompting 
def execute_with_retry(model: str, ticket: str, prompt_version: str, max_retries: int = 1) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    system_prompt = PROMPTS.get(prompt_version, PROMPTS.get(ACTIVE_VERSION))
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Ticket:\n{ticket}"}
    ]
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        resp = ollama.chat(model=model, messages=messages, options={"temperature": 0})
        raw_output = resp["message"]["content"]
        
        data, error = parse_and_validate(raw_output)
        
        if data is not None:
            return data, attempt, None # Success
            
        # Reprompt logic: Feed the exact Pydantic error back to the model
        last_error = error
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({"role": "user", "content": f"INVALID OUTPUT. Error: {error}. Fix this and return ONLY valid JSON matching the schema."})
        
    return None, max_retries, last_error # Failed completely

# --- Endpoints ---
@app.post("/api/v1/extract")
async def process_ticket(req: TicketRequest):
    requested_version = req.prompt_version if req.prompt_version in PROMPTS else ACTIVE_VERSION
    
    # 1. Primary Pass (Includes internal reprompt loop)
    data, retries, primary_error = execute_with_retry(PRIMARY_MODEL, req.ticket_text, requested_version)
    
    # 2. Heuristic Routing Trigger
    needs_fallback = False
    ticket_lower = req.ticket_text.lower()
    
    if data is None:
        # Schema validation failed completely
        needs_fallback = True
    elif data.get("email") is None and data.get("refund_amount") is None:
        # Failed to find core business data
        needs_fallback = True
    elif data.get("name") is None and data.get("email") is not None:
        # Suspicious: Found an email but missed the name. 1.5B often fails this.
        needs_fallback = True
    elif data.get("refund_amount") is not None and data.get("currency") is None:
        # Suspicious: Found a number but missed the currency.
        needs_fallback = True
    elif ">" in req.ticket_text or "forwarded" in ticket_lower or "---" in req.ticket_text:
        # Known limitation: 1.5B models fail temporal context and quoted replies.
        needs_fallback = True
    # 3. Fallback Cascade (Includes its own internal reprompt loop)
    if needs_fallback:
        print(f"[Router] {PRIMARY_MODEL} failed or returned empty data. Escalating to {FALLBACK_MODEL}...")
        fb_data, fb_retries, fb_error = execute_with_retry(FALLBACK_MODEL, req.ticket_text, requested_version)
        
        if fb_data is None:
            raise HTTPException(
                status_code=422, 
                detail=f"Pipeline Failure. Primary error: {primary_error}. Fallback error: {fb_error}"
            )
            
        return {
            "status": "success",
            "model_used": FALLBACK_MODEL,
            "prompt_version": requested_version,
            "internal_retries": fb_retries,
            "data": fb_data
        }

    return {
        "status": "success",
        "model_used": PRIMARY_MODEL,
        "prompt_version": requested_version,
        "internal_retries": retries,
        "data": data
    }

@app.get("/health")
async def health_check():
    return {
        "status": "operational", 
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "active_prompt": ACTIVE_VERSION,
        "available_prompts": list(PROMPTS.keys())
    }
