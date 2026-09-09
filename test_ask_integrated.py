import subprocess
import time
import requests
import sys

print("Starting server subprocess...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

time.sleep(5)  # wait for uvicorn to start

try:
    print("Checking /health...")
    res = requests.get("http://127.0.0.1:8000/health", timeout=3)
    print("Health:", res.status_code, res.json())
    
    print("\nSending /ask...")
    body = {
        "question": "Show an ER diagram for Customers and Orders",
        "history": []
    }
    res = requests.post("http://127.0.0.1:8000/ask", json=body, timeout=60)
    print("Ask Status:", res.status_code)
    
    if res.status_code == 200:
        data = res.json()
        print("Response text length:", len(data.get("response", "")))
        print("Visual Type:", data.get("visual_type"))
        print("Visual Required:", data.get("visual_required"))
        seq = data.get("teaching_sequence", [])
        print(f"Teaching Steps: {len(seq)}")
        for step in seq:
            print(f" - {step.get('step_id')}: audio={step.get('audio_url')}, duration={step.get('duration_hint')}")
    else:
        print("Error content:", res.text)
        
except Exception as e:
    print("Test Exception:", e)
finally:
    print("\n--- Server Logs ---")
    proc.terminate()
    out, _ = proc.communicate(timeout=5)
    print(out)
