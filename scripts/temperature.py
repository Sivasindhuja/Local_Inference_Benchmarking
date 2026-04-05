import ollama
import json

def test_stochasticity(model_name, concept, temperature):
    print(f"\n--- Testing Temp {temperature} for: {concept} ---")
    explanations = []
    
    for i in range(5):
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': "Output JSON: {'concept': str, 'explanation': str}"},
                {'role': 'user', 'content': f"Explain {concept}"}
            ],
            format="json",
            options={'temperature': temperature}
        )
        data = json.loads(response['message']['content'])
        explanations.append(data['explanation'])
        print(f"Run {i+1} completed.")

    # Check for variance
    unique_count = len(set(explanations))
    print(f">> Unique Explanations: {unique_count}/5")
    return explanations

# Run the test
llama_temp_0 = test_stochasticity("mistral", "Docker Volumes", 0)#11:25-11:29
llama_temp_07 = test_stochasticity("mistral", "Docker Volumes", 0.7)