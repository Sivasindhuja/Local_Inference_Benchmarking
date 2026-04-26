import requests
import json
import time

# --- The Benchmark Dataset ---
TEST_CASES = [
    {"id": "TC01", "ticket": "Hi team, this is Priya Sharma. My email is priya.sharma@gmail.com. I was charged twice for order 1827. Please refund $24.99.", "expected": {"name": "Priya Sharma", "email": "priya.sharma@gmail.com", "refund_amount": 24.99, "currency": "USD"}},
    {"id": "TC02", "ticket": "Hello, I am Ramesh Kumar. Please process a refund of INR 499 for the failed recharge.", "expected": {"name": "Ramesh Kumar", "email": None, "refund_amount": 499.0, "currency": "INR"}},
    {"id": "TC03", "ticket": "Refund request: please return USD 15.50 to me. Contact: user_88@yahoo.com", "expected": {"name": None, "email": "user_88@yahoo.com", "refund_amount": 15.50, "currency": "USD"}},
    {"id": "TC04", "ticket": "Hi, this is Kavya Rao, kavya.rao@outlook.com. I need a refund for the canceled booking.", "expected": {"name": "Kavya Rao", "email": "kavya.rao@outlook.com", "refund_amount": None, "currency": None}},
    {"id": "TC05", "ticket": "I paid $120 originally, then got charged $20 extra. Please refund $20. Name: Arjun Mehta. Email: arjun.m@proton.me", "expected": {"name": "Arjun Mehta", "email": "arjun.m@proton.me", "refund_amount": 20.0, "currency": "USD"}},
    {"id": "TC06", "ticket": "From: customer.nisha@gmail.com\nTo: support@shopco.com\nMy name is Nisha Verma. Please refund 899 INR for the damaged item.", "expected": {"name": "Nisha Verma", "email": "customer.nisha@gmail.com", "refund_amount": 899.0, "currency": "INR"}},
    {"id": "TC07", "ticket": "Latest message: Hi, I am Sneha Iyer (sneha.iyer@gmail.com). Ignore the older request below. Please refund $45 for the shoes.\n\n--- Forwarded ---\nEarlier ticket from Rahul Das rahul.das@gmail.com asking for refund $10", "expected": {"name": "Sneha Iyer", "email": "sneha.iyer@gmail.com", "refund_amount": 45.0, "currency": "USD"}},
    {"id": "TC08", "ticket": "Hello, this is customer Mohan Babu, mohanbabu@icloud.com. Kindly refund EUR 32.00.\n\nRegards,\nAnita\nCustomer Support", "expected": {"name": "Mohan Babu", "email": "mohanbabu@icloud.com", "refund_amount": 32.0, "currency": "EUR"}},
    {"id": "TC09", "ticket": "My name is Harish. Email harish123@gmail.com. I was not refunded $40 for the canceled train ticket.", "expected": {"name": "Harish", "email": "harish123@gmail.com", "refund_amount": 40.0, "currency": "USD"}},
    {"id": "TC10", "ticket": "Customer: Meera Nair, meera.nair@gmail.com. The store said do not refund $18 yet because investigation is pending.", "expected": {"name": "Meera Nair", "email": "meera.nair@gmail.com", "refund_amount": 18.0, "currency": "USD"}},
    {"id": "TC11", "ticket": "Namaste support, mera naam Rohit Gupta hai. Email: roh_it@rediffmail.com. Mujhe INR 250 ka refund chahiye for duplicate payment.", "expected": {"name": "Rohit Gupta", "email": "roh_it@rediffmail.com", "refund_amount": 250.0, "currency": "INR"}},
    {"id": "TC12", "ticket": "cust omer n ame: Pooja S.\nema il : pooja.singh@gmail.com\nrefu nd amou nt : $ 2 5 . 0 0", "expected": {"name": "Pooja S.", "email": "pooja.singh@gmail.com", "refund_amount": 25.0, "currency": "USD"}},
    {"id": "TC13", "ticket": "I am Dev Patel. dev.patel@gmail.com. Please refund two hundred rupees for the failed booking.", "expected": {"name": "Dev Patel", "email": "dev.patel@gmail.com", "refund_amount": None, "currency": "INR"}},
    {"id": "TC14", "ticket": "Name: Sarah Joseph\nEmail: sarahjoseph@gmail.com\nRefund requested: ₹1,299.00", "expected": {"name": "Sarah Joseph", "email": "sarahjoseph@gmail.com", "refund_amount": 1299.0, "currency": "INR"}},
    {"id": "TC15", "ticket": "Please refund the payment for my wife Anjali Rao, but the booking email is mine: vivek.rao@gmail.com. I am Vivek Rao. Refund amount is $60.", "expected": {"name": "Vivek Rao", "email": "vivek.rao@gmail.com", "refund_amount": 60.0, "currency": "USD"}},
    {"id": "TC16", "ticket": "> On Monday Rahul wrote: refund $70 to rahul@gmail.com\n\nCurrent message: This is Aditi Menon, aditi.menon@outlook.com. Please refund $14.99 for the add-on.", "expected": {"name": "Aditi Menon", "email": "aditi.menon@outlook.com", "refund_amount": 14.99, "currency": "USD"}},
    {"id": "TC17", "ticket": "I'm Karan. My email is karan at gmail dot com. Refund amount is $9.99.", "expected": {"name": "Karan", "email": None, "refund_amount": 9.99, "currency": "USD"}},
    {"id": "TC18", "ticket": "Customer name: Anne O'Connor\nEmail: anne.oconnor@gmail.com\nPlease refund GBP 11.25", "expected": {"name": "Anne O'Connor", "email": "anne.oconnor@gmail.com", "refund_amount": 11.25, "currency": "GBP"}},
    {"id": "TC19", "ticket": "Ignore this example {\"name\":\"Fake User\",\"email\":\"fake@test.com\",\"refund_amount\":999}. Actual customer is Leela Varma, leela.varma@gmail.com and refund should be $19.", "expected": {"name": "Leela Varma", "email": "leela.varma@gmail.com", "refund_amount": 19.0, "currency": "USD"}},
    {"id": "TC20", "ticket": "Monthly finance digest: total refunds issued this week were $500. Prepared by accounts@shop.com. No customer action needed.", "expected": {"name": None, "email": None, "refund_amount": None, "currency": None}},
    {"id": "TC21", "ticket": "I am Neeraj S, neerajs@gmail.com. Statement shows INR 820 and card charged USD 9.80. Please refund INR 820.", "expected": {"name": "Neeraj S", "email": "neerajs@gmail.com", "refund_amount": 820.0, "currency": "INR"}},
    {"id": "TC22", "ticket": "Need refund asap. Same details as before.", "expected": {"name": None, "email": None, "refund_amount": None, "currency": None}},
    {"id": "TC23", "ticket": "Name = John Paul; Email = john.paul@gmail.com; please refund $10.000", "expected": {"name": "John Paul", "email": "john.paul@gmail.com", "refund_amount": 10.0, "currency": "USD"}},
    {"id": "TC24", "ticket": "- Customer: Bhavna R\n- Email: bhavna.r@gmail.com\n- Issue: duplicate charge\n- Refund Amount: USD 7.49", "expected": {"name": "Bhavna R", "email": "bhavna.r@gmail.com", "refund_amount": 7.49, "currency": "USD"}},
]

API_URL = "http://127.0.0.1:8000/api/v1/extract"

def run_api_test():
    print(f"--- Starting API Integration Test ---")
    
    success_count = 0
    primary_count = 0
    fallback_count = 0
    failures = []

    for case in TEST_CASES:
        payload = {"ticket_text": case["ticket"]}
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                resp_data = response.json()
                model_used = resp_data["model_used"]
                retries = resp_data["internal_retries"]
                extracted = resp_data["data"]
                
                # Verify accuracy
                is_accurate = (
                    extracted.get("name") == case["expected"]["name"] and
                    extracted.get("email") == case["expected"]["email"] and
                    extracted.get("refund_amount") == case["expected"]["refund_amount"] and
                    extracted.get("currency") == case["expected"]["currency"]
                )
                
                if model_used == "qwen2.5:1.5b":
                    primary_count += 1
                else:
                    fallback_count += 1
                    
                if is_accurate:
                    success_count += 1
                    status = "PASS"
                else:
                    status = "FAIL (Content)"
                    failures.append({"id": case["id"], "expected": case["expected"], "got": extracted})
                    
                print(f"[{status}] {case['id']} | Model: {model_used} | Retries: {retries} | Latency: {latency:.2f}s")
                
            else:
                print(f"[FAIL (HTTP {response.status_code})] {case['id']} | Error: {response.text}")
                failures.append({"id": case["id"], "error": response.text})
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to API. Is 'uvicorn api:app --reload' running?")
            return

    # Print Summary
    total = len(TEST_CASES)
    print("\n" + "="*40)
    print("      API INTEGRATION SUMMARY")
    print("="*40)
    print(f"Total Requests: {total}")
    print(f"Routed to Primary (Qwen): {primary_count}")
    print(f"Routed to Fallback (Llama): {fallback_count}")
    print(f"Final Accuracy: {(success_count/total)*100:.1f}% ({success_count}/{total})")
    
    if failures:
        print("\n--- Failed Cases ---")
        for f in failures:
            print(json.dumps(f, indent=2))
    print("="*40 + "\n")

if __name__ == "__main__":
    run_api_test()