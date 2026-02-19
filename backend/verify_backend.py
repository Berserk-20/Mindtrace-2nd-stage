import requests
import json

base_url = "http://127.0.0.1:8001"

def test():
    # 1. Login or Signup
    email = "testVal@example.com"
    password = "TestPassword123"
    
    print(f"Attempting login for {email}...")
    try:
        resp = requests.post(f"{base_url}/login", json={"email": email, "password": password})
        if resp.status_code == 401:
            print("Login failed, attempting signup...")
            signup_resp = requests.post(f"{base_url}/signup", json={
                "name": "Test User",
                "email": email,
                "password": password
            })
            print(f"Signup status: {signup_resp.status_code}")
            if signup_resp.status_code not in [200, 400]: # 400 if already exists but wrong password?
                print(f"Signup failed: {signup_resp.text}")
                return
                
            # Retry login
            resp = requests.post(f"{base_url}/login", json={"email": email, "password": password})
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    if resp.status_code != 200:
        print(f"Login failed: {resp.status_code} {resp.text}")
        return

    token = resp.json().get("token")
    if not token:
        print("No token in login response")
        return
        
    print("Login successful, got token.")
    
    # 2. Call /metrics
    print("Calling /metrics...")
    headers = {"Authorization": f"Bearer {token}"}
    metrics_resp = requests.get(f"{base_url}/metrics", headers=headers)
    
    print(f"Metrics Status: {metrics_resp.status_code}")
    if metrics_resp.status_code == 200:
        print("Metrics response OK")
        # Optional: Print keys to verify structure
        data = metrics_resp.json()
        print("Keys:", list(data.keys()))
    else:
        print(f"Metrics failed: {metrics_resp.text}")

if __name__ == "__main__":
    test()
