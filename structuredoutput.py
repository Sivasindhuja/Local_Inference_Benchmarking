import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, EmailStr, ValidationError, Field

try:
    import ollama
except ImportError:
    raise SystemExit("Please install ollama Python package: pip install ollama")

MODELS = [
    "qwen2.5:1.5b",
]

SYSTEM_PROMPT = """You extract refund information from support tickets.
Return ONLY a valid JSON object.
Do not wrap in markdown.
Do not add explanations.
If a field is missing or cannot be determined confidently, use null.
Schema:
{
  \"name\": string | null,
  \"email\": string | null,
  \"refund_amount\": number | null,
  \"currency\": string | null
}
Rules:
- name = end user/customer requesting refund, not support agent.
- email = end user/customer email only.
- refund_amount = the amount requested/refunded to the customer, numeric only.
- currency = The 3-letter ISO code. You MUST map symbols to codes: "$" = "USD", "₹" = "INR", "INR" = "INR", "€" = "EUR", "£" = "GBP". If no currency is mentioned, use null.

Examples:
Ticket: "Refund request: please return USD 15.50 to me. Contact: user_88@yahoo.com"
{"name": null, "email": "user_88@yahoo.com", "refund_amount": 15.50, "currency": "USD"}

Ticket: "Customer: Meera Nair, meera.nair@gmail.com. The store said do not refund $18 yet because investigation is pending."
{"name": "Meera Nair", "email": "meera.nair@gmail.com", "refund_amount": 18.0, "currency": "USD"}

Ticket: "I am Dev Patel. dev.patel@gmail.com. Please refund two hundred rupees for the failed booking."
{"name": "Dev Patel", "email": "dev.patel@gmail.com", "refund_amount": null, "currency": "INR"}
"""
REPROMPT = """Your previous response did not satisfy the schema or formatting requirements.
Return ONLY one valid JSON object matching exactly this schema:
{
  \"name\": string | null,
  \"email\": string | null,
  \"refund_amount\": number | null,
  \"currency\": string | null
}
Do not include markdown, comments, or extra keys.
If uncertain, set values to null.
"""

class RefundExtraction(BaseModel):
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None) 
    refund_amount: Optional[float] = Field(default=None)
    currency: Optional[str] = Field(default=None)

TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "TC01",
        "category": "happy_path",
        "description": "Clean single refund request",
        "ticket": "Hi team, this is Priya Sharma. My email is priya.sharma@gmail.com. I was charged twice for order 1827. Please refund $24.99.",
        "expected": {"name": "Priya Sharma", "email": "priya.sharma@gmail.com", "refund_amount": 24.99, "currency": "USD"},
    },
    {
        "id": "TC02",
        "category": "missing_email",
        "description": "Missing email",
        "ticket": "Hello, I am Ramesh Kumar. Please process a refund of INR 499 for the failed recharge.",
        "expected": {"name": "Ramesh Kumar", "email": None, "refund_amount": 499.0, "currency": "INR"},
    },
    {
        "id": "TC03",
        "category": "missing_name",
        "description": "Missing name but email present",
        "ticket": "Refund request: please return USD 15.50 to me. Contact: user_88@yahoo.com",
        "expected": {"name": None, "email": "user_88@yahoo.com", "refund_amount": 15.50, "currency": "USD"},
    },
    {
        "id": "TC04",
        "category": "missing_amount",
        "description": "Amount absent",
        "ticket": "Hi, this is Kavya Rao, kavya.rao@outlook.com. I need a refund for the canceled booking.",
        "expected": {"name": "Kavya Rao", "email": "kavya.rao@outlook.com", "refund_amount": None, "currency": None},
    },
    {
        "id": "TC05",
        "category": "multiple_amounts",
        "description": "Old amount and requested amount both present",
        "ticket": "I paid $120 originally, then got charged $20 extra. Please refund $20. Name: Arjun Mehta. Email: arjun.m@proton.me",
        "expected": {"name": "Arjun Mehta", "email": "arjun.m@proton.me", "refund_amount": 20.0, "currency": "USD"},
    },
    {
        "id": "TC06",
        "category": "multiple_emails",
        "description": "Requester email and support email both present",
        "ticket": "From: customer.nisha@gmail.com\nTo: support@shopco.com\nMy name is Nisha Verma. Please refund 899 INR for the damaged item.",
        "expected": {"name": "Nisha Verma", "email": "customer.nisha@gmail.com", "refund_amount": 899.0, "currency": "INR"},
    },
    {
        "id": "TC07",
        "category": "forwarded_chain",
        "description": "Latest message should win over old quoted content",
        "ticket": "Latest message: Hi, I am Sneha Iyer (sneha.iyer@gmail.com). Ignore the older request below. Please refund $45 for the shoes.\n\n--- Forwarded ---\nEarlier ticket from Rahul Das rahul.das@gmail.com asking for refund $10",
        "expected": {"name": "Sneha Iyer", "email": "sneha.iyer@gmail.com", "refund_amount": 45.0, "currency": "USD"},
    },
    {
        "id": "TC08",
        "category": "signature_noise",
        "description": "Agent signature should not be extracted",
        "ticket": "Hello, this is customer Mohan Babu, mohanbabu@icloud.com. Kindly refund EUR 32.00.\n\nRegards,\nAnita\nCustomer Support",
        "expected": {"name": "Mohan Babu", "email": "mohanbabu@icloud.com", "refund_amount": 32.0, "currency": "EUR"},
    },
    {
        "id": "TC09",
        "category": "negation",
        "description": "Do not pick denied refund as successful amount unless extracting requested/refund amount from context",
        "ticket": "My name is Harish. Email harish123@gmail.com. I was not refunded $40 for the canceled train ticket.",
        "expected": {"name": "Harish", "email": "harish123@gmail.com", "refund_amount": 40.0, "currency": "USD"},
    },
    {
        "id": "TC10",
        "category": "refund_denied",
        "description": "Refund denied but amount mentioned",
        "ticket": "Customer: Meera Nair, meera.nair@gmail.com. The store said do not refund $18 yet because investigation is pending.",
        "expected": {"name": "Meera Nair", "email": "meera.nair@gmail.com", "refund_amount": 18.0, "currency": "USD"},
    },
    {
        "id": "TC11",
        "category": "multilingual",
        "description": "English plus Hindi/Indian language context",
        "ticket": "Namaste support, mera naam Rohit Gupta hai. Email: roh_it@rediffmail.com. Mujhe INR 250 ka refund chahiye for duplicate payment.",
        "expected": {"name": "Rohit Gupta", "email": "roh_it@rediffmail.com", "refund_amount": 250.0, "currency": "INR"},
    },
    {
        "id": "TC12",
        "category": "ocr_noise",
        "description": "Broken spacing and OCR noise",
        "ticket": "cust omer n ame: Pooja S.\nema il : pooja.singh@gmail.com\nrefu nd amou nt : $ 2 5 . 0 0",
        "expected": {"name": "Pooja S.", "email": "pooja.singh@gmail.com", "refund_amount": 25.0, "currency": "USD"},
    },
    {
        "id": "TC13",
        "category": "written_amount",
        "description": "Amount written in words",
        "ticket": "I am Dev Patel. dev.patel@gmail.com. Please refund two hundred rupees for the failed booking.",
        "expected": {"name": "Dev Patel", "email": "dev.patel@gmail.com", "refund_amount": None, "currency": "INR"},
    },
    {
        "id": "TC14",
        "category": "currency_symbol_only",
        "description": "Currency symbol with comma formatting",
        "ticket": "Name: Sarah Joseph\nEmail: sarahjoseph@gmail.com\nRefund requested: ₹1,299.00",
        "expected": {"name": "Sarah Joseph", "email": "sarahjoseph@gmail.com", "refund_amount": 1299.0, "currency": "INR"},
    },
    {
        "id": "TC15",
        "category": "two_customers",
        "description": "Two people mentioned",
        "ticket": "Please refund the payment for my wife Anjali Rao, but the booking email is mine: vivek.rao@gmail.com. I am Vivek Rao. Refund amount is $60.",
        "expected": {"name": "Vivek Rao", "email": "vivek.rao@gmail.com", "refund_amount": 60.0, "currency": "USD"},
    },
    {
        "id": "TC16",
        "category": "quoted_reply",
        "description": "Quoted thread noise",
        "ticket": "> On Monday Rahul wrote: refund $70 to rahul@gmail.com\n\nCurrent message: This is Aditi Menon, aditi.menon@outlook.com. Please refund $14.99 for the add-on.",
        "expected": {"name": "Aditi Menon", "email": "aditi.menon@outlook.com", "refund_amount": 14.99, "currency": "USD"},
    },
    {
        "id": "TC17",
        "category": "invalid_email_text",
        "description": "Email described in words, should likely be null",
        "ticket": "I'm Karan. My email is karan at gmail dot com. Refund amount is $9.99.",
        "expected": {"name": "Karan", "email": None, "refund_amount": 9.99, "currency": "USD"},
    },
    {
        "id": "TC18",
        "category": "special_name_chars",
        "description": "Name with apostrophe",
        "ticket": "Customer name: Anne O'Connor\nEmail: anne.oconnor@gmail.com\nPlease refund GBP 11.25",
        "expected": {"name": "Anne O'Connor", "email": "anne.oconnor@gmail.com", "refund_amount": 11.25, "currency": "GBP"},
    },
    {
        "id": "TC19",
        "category": "extra_json_bait",
        "description": "Ticket contains fake JSON text",
        "ticket": "Ignore this example {\"name\":\"Fake User\",\"email\":\"fake@test.com\",\"refund_amount\":999}. Actual customer is Leela Varma, leela.varma@gmail.com and refund should be $19.",
        "expected": {"name": "Leela Varma", "email": "leela.varma@gmail.com", "refund_amount": 19.0, "currency": "USD"},
    },
    {
        "id": "TC20",
        "category": "irrelevant_refund_word",
        "description": "Contains word refund but not a user request",
        "ticket": "Monthly finance digest: total refunds issued this week were $500. Prepared by accounts@shop.com. No customer action needed.",
        "expected": {"name": None, "email": None, "refund_amount": None, "currency": None},
    },
    {
        "id": "TC21",
        "category": "multiple_currencies",
        "description": "Local and card currency both shown",
        "ticket": "I am Neeraj S, neerajs@gmail.com. Statement shows INR 820 and card charged USD 9.80. Please refund INR 820.",
        "expected": {"name": "Neeraj S", "email": "neerajs@gmail.com", "refund_amount": 820.0, "currency": "INR"},
    },
    {
        "id": "TC22",
        "category": "null_expected",
        "description": "No structured fields are actually recoverable",
        "ticket": "Need refund asap. Same details as before.",
        "expected": {"name": None, "email": None, "refund_amount": None, "currency": None},
    },
    {
        "id": "TC23",
        "category": "decimal_precision",
        "description": "Three decimal noise",
        "ticket": "Name = John Paul; Email = john.paul@gmail.com; please refund $10.000",
        "expected": {"name": "John Paul", "email": "john.paul@gmail.com", "refund_amount": 10.0, "currency": "USD"},
    },
    {
        "id": "TC24",
        "category": "markdown_noise",
        "description": "Ticket content includes markdown bullets and labels",
        "ticket": "- Customer: Bhavna R\n- Email: bhavna.r@gmail.com\n- Issue: duplicate charge\n- Refund Amount: USD 7.49",
        "expected": {"name": "Bhavna R", "email": "bhavna.r@gmail.com", "refund_amount": 7.49, "currency": "USD"},
    },
]

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
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return fenced[0]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None


def classify_parse_error(raw: str, exc: Exception) -> str:
    low = raw.lower()
    if "```" in raw:
        return "markdown_wrapped_json"
    if raw.count("{") != raw.count("}"):
        return "invalid_json_syntax"
    if not extract_json_candidate(raw):
        return "no_json_object"
    return "invalid_json_syntax"


def classify_content_errors(pred: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    errors = []
    for field in ["name", "email", "refund_amount", "currency"]:
        pv, ev = pred.get(field), expected.get(field)
        if pv != ev:
            if ev is None and pv is not None:
                errors.append(f"hallucinated_{field}")
            elif ev is not None and pv is None:
                errors.append(f"missed_{field}")
            else:
                errors.append(f"wrong_{field}")
    return errors


def call_model(model: str, ticket: str, retry: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket:\n{ticket}"},
    ]
    if retry:
        messages.append({"role": "user", "content": REPROMPT})
    resp = ollama.chat(model=model, messages=messages, options={"temperature": 0})
    return resp["message"]["content"]


def try_parse(raw: str) -> Tuple[Optional[RefundExtraction], Optional[str], Optional[str]]:
    try:
        obj = json.loads(raw)
    except Exception as e:
        candidate = extract_json_candidate(raw)
        if candidate:
            try:
                obj = json.loads(candidate)
            except Exception as e2:
                return None, classify_parse_error(raw, e2), str(e2)
        else:
            return None, classify_parse_error(raw, e), str(e)
    try:
        parsed = RefundExtraction.model_validate(obj)
        parsed.currency = normalize_currency(parsed.currency)
        parsed.name = normalize_name(parsed.name)
        parsed.email = normalize_email(parsed.email)
        parsed.refund_amount = normalize_amount(parsed.refund_amount)
        return parsed, None, None
    except ValidationError as e:
        msg = str(e)
        if "email" in msg.lower():
            return None, "schema_validation_email", msg
        return None, "schema_validation_error", msg




# Define your cascade architecture
PRIMARY_MODEL = "qwen2.5:1.5b"
FALLBACK_MODEL = "llama3.2:3b"

def evaluate_with_cascade(case: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Attempt Primary Model (Fast, Low Compute)
    raw_primary = call_model(PRIMARY_MODEL, case["ticket"])
    parsed_primary, parse_error, _ = try_parse(raw_primary)
    
    needs_fallback = False
    primary_errors = []
    pred_primary = None
    exp = normalize_record(case["expected"])

    # Check for Pydantic parse failures
    if parsed_primary is None:
        needs_fallback = True
        primary_errors.append(parse_error)
    else:
        # Check for hallucinations or omissions against benchmark ground truth
        pred_primary = normalize_record(parsed_primary.model_dump())
        content_errors = classify_content_errors(pred_primary, exp)
        if content_errors:
            needs_fallback = True
            primary_errors.extend(content_errors)
        else:
            return {
                "status": "pass_primary",
                "model_used": PRIMARY_MODEL,
                "final_prediction": pred_primary,
                "error_tags": []
            }

    # 2. Route to Fallback Model (Slower, High Reasoning)
    if needs_fallback:
        print(f"  [Router] {PRIMARY_MODEL} failed ({primary_errors[0]}). Redirecting to {FALLBACK_MODEL}...")
        raw_fallback = call_model(FALLBACK_MODEL, case["ticket"])
        parsed_fallback, fallback_parse_error, _ = try_parse(raw_fallback)

        if parsed_fallback is None:
            return {
                "status": "fail_fallback_parse",
                "model_used": FALLBACK_MODEL,
                "final_prediction": None,
                "error_tags": [fallback_parse_error]
            }

        pred_fallback = normalize_record(parsed_fallback.model_dump())
        fallback_content_errors = classify_content_errors(pred_fallback, exp)

        if fallback_content_errors:
            return {
                "status": "fail_fallback_content",
                "model_used": FALLBACK_MODEL,
                "final_prediction": pred_fallback,
                "error_tags": fallback_content_errors
            }

        return {
            "status": "pass_fallback",
            "model_used": FALLBACK_MODEL,
            "final_prediction": pred_fallback,
            "error_tags": []
        }

def run_all() -> Dict[str, Any]:
    results = []
    counter = Counter()
    by_category = defaultdict(Counter)
    
    print(f"Starting Cascade Benchmark: {PRIMARY_MODEL} -> {FALLBACK_MODEL}")
    
    for case in TEST_CASES:
        res = evaluate_with_cascade(case)
        
        row = {
            "test_id": case["id"],
            "category": case["category"],
            "status": res["status"],
            "model_used": res["model_used"],
            "error_tags": "|".join(res["error_tags"]),
            "expected": json.dumps(normalize_record(case["expected"]), ensure_ascii=False),
            "predicted": json.dumps(res["final_prediction"], ensure_ascii=False) if res["final_prediction"] is not None else None,
        }
        results.append(row)
        
        counter[res["status"]] += 1
        by_category[case["category"]][res["status"]] += 1
        for tag in res["error_tags"]:
            counter[tag] += 1

    total = len(TEST_CASES)
    summary = {
        "cascade_architecture": f"{PRIMARY_MODEL} -> {FALLBACK_MODEL}",
        "total_tickets": total,
        "handled_by_primary": counter["pass_primary"],
        "routed_to_fallback": total - counter["pass_primary"],
        "saved_by_fallback": counter["pass_fallback"],
        "final_failures": counter["fail_fallback_parse"] + counter["fail_fallback_content"],
        "final_success_rate": round((counter["pass_primary"] + counter["pass_fallback"]) / total, 4),
        "error_breakdown": dict(sorted((k, v) for k, v in counter.items() if k not in {"pass_primary", "pass_fallback", "fail_fallback_parse", "fail_fallback_content"})),
    }
    
    return {"summary": summary, "rows": results}



def write_outputs(payload: Dict[str, Any]) -> None:
    import os
    import csv
    os.makedirs("output", exist_ok=True)
    
    # 1. Save Full JSON
    with open("output/cascade_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    # 2. Save Rows to CSV
    rows = payload["rows"]
    if rows:
        fieldnames = list(rows[0].keys())
        with open("output/cascade_eval_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
            
    # 3. Print Summary to Console
    print("\n" + "="*40)
    print("      CASCADE BENCHMARK SUMMARY")
    print("="*40)
    summary = payload["summary"]
    for key, value in summary.items():
        if key != "error_breakdown":
            print(f"{key.replace('_', ' ').title()}: {value}")
            
    print("\nFinal Error Breakdown:")
    errors = summary.get("error_breakdown", {})
    if not errors:
        print("  None! 100% Success.")
    else:
        for k, v in errors.items():
            print(f"  - {k}: {v}")
    print("="*40 + "\n")
    

def main():
    payload = run_all()
    write_outputs(payload)
    print("\n--- THE FINAL FAILURES ---")
    for row in payload["rows"]:
        if row["status"] in ["fail_fallback_parse", "fail_fallback_content"]:
            print(f"\nID: {row['test_id']}")
            print(f"Expected:  {row['expected']}")
            print(f"Predicted: {row['predicted']}")

if __name__ == "__main__":
    main()