import sys
import os

# Add current directory to path so we can import from db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import create_user, user_exists, users_col
from datetime import datetime

ADMIN_EMAIL = "gaikwadsankalp11@gmail.com"
ADMIN_PASSWORD = "Admin@123"
ADMIN_NAME = "Admin"

def create_admin_user():
    print(f"Checking if admin user {ADMIN_EMAIL} exists...")
    
    if user_exists(ADMIN_EMAIL):
        print("Admin user already exists.")
        # Optional: Update role if it exists but isn't admin
        user = users_col.find_one({"email": ADMIN_EMAIL})
        if user.get("role") != "admin":
            print("Updating existing user to admin role...")
            users_col.update_one({"email": ADMIN_EMAIL}, {"$set": {"role": "admin"}})
            print("Role updated.")
        return

    print("Creating admin user...")
    try:
        create_user(ADMIN_NAME, ADMIN_EMAIL, ADMIN_PASSWORD, role="admin")
        print("Admin user created successfully!")
    except Exception as e:
        print(f"Error creating admin user: {e}")

if __name__ == "__main__":
    create_admin_user()
