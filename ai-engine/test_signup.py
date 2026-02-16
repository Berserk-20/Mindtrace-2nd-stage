import requests
import json
import time

try:
    print("Testing Signup...")
    url = "http://127.0.0.1:8000/signup"
    payload = {
        "name": "admin",
        "email": "admin123@gmail.com",
        "password": "password"
    }
    headers = {"Content-Type": "application/json"}
    
    start = time.time()
    response = requests.post(url, json=payload, timeout=5)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200 or "Email already registered" in response.text:
        print("Signup test passed (or already exists)")
    else:
        print("Signup test failed")

except Exception as e:
    print(f"Error: {e}")
