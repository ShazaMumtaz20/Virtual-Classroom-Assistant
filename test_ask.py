import requests
import json

try:
    print("Sending /health request...")
    res_health = requests.get("http://127.0.0.1:8000/health", timeout=3)
    print("Health:", res_health.json())
    
    print("\nSending /ask request...")
    body = {
        "question": "Explain an ER relationship between Customers and Orders",
        "history": []
    }
    res_ask = requests.post("http://127.0.0.1:8000/ask", json=body, timeout=45)
    
    if res_ask.status_code == 200:
        data = res_ask.json()
        print("\nResponse:", data.get("response")[:100], "...")
        print("Visual Type:", data.get("visual_type"))
        print("Visual Required:", data.get("visual_required"))
        seq = data.get("teaching_sequence", [])
        print(f"Teaching Sequence steps: {len(seq)}")
        if seq:
            print("First Step Audio URL:", seq[0].get("audio_url"))
    else:
        print("\nError:", res_ask.status_code, res_ask.text)
        
except Exception as e:
    print("Exception occurred:", str(e))
