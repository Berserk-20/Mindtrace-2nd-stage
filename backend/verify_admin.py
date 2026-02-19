
import requests
import sys

BASE_URL = "http://localhost:8000"

def test_admin_access():
    print("Testing Admin Access...")
    # 1. Login as Admin
    login_data = {"email": "gaikwadsankalp11@gmail.com", "password": "Admin@123"}
    try:
        res = requests.post(f"{BASE_URL}/login", json=login_data)
        if res.status_code != 200:
            print(f"FAILED: Admin login failed. Status: {res.status_code}, Response: {res.text}")
            return
        
        token = res.json()["token"]
        print("Admin Logged in successfully.")

        # 2. Access Admin Stats
        headers = {"Authorization": f"Bearer {token}"}
        stats_res = requests.get(f"{BASE_URL}/api/admin/stats", headers=headers)
        
        if stats_res.status_code == 200:
            print("SUCCESS: Admin accessed stats.")
            print("Stats:", stats_res.json()["stats"])
        else:
            print(f"FAILED: Admin could not access stats. Status: {stats_res.status_code}, Response: {stats_res.text}")

    except Exception as e:
        print(f"ERROR: {e}")

def test_user_access():
    print("\nTesting Regular User Access...")
    # 1. Create/Login as Regular User
    # For now, let's try to login as a user if one exists, or just fail if we can't.
    # Actually, let's create a temp user using the API if possible, or assume one exists.
    # Since I don't reused a known user password, I'll rely on the existing 'signup' or just create one.
    
    # Let's try to signup a test user
    test_email = "test_user_verify@example.com"
    test_pass = "User@123"
    
    try:
        # Try signup
        requests.post(f"{BASE_URL}/signup", json={"name": "Test User", "email": test_email, "password": test_pass})
        
        # Login
        res = requests.post(f"{BASE_URL}/login", json={"email": test_email, "password": test_pass})
        if res.status_code != 200:
            print("Skipping User test: Could not login as test user.")
            return

        token = res.json()["token"]
        print("Regular User Logged in.")

        # 2. Access Admin Stats (Should Fail)
        headers = {"Authorization": f"Bearer {token}"}
        stats_res = requests.get(f"{BASE_URL}/api/admin/stats", headers=headers)
        
        if stats_res.status_code == 403:
            print("SUCCESS: Regular User denied access (403).")
        else:
            print(f"FAILED: Regular User was NOT denied access. Status: {stats_res.status_code}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    try:
        test_admin_access()
        test_user_access()
    except Exception as e:
        print(f"Critical Error: {e}")
        print("Is the backend running?")
