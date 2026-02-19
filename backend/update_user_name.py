
import sys
import os

# Add current directory to path so we can import from db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import users_col

EMAIL = "sankalpgaikwad10@gmail.com"
NEW_NAME = "Sankalp Gaikwad"

def update_user_name():
    print(f"Checking if user {EMAIL} exists...")
    
    user = users_col.find_one({"email": EMAIL})
    
    if user:
        print(f"User found. Current name: {user.get('name')}")
        print(f"Updating name to: {NEW_NAME}")
        
        result = users_col.update_one(
            {"email": EMAIL},
            {"$set": {"name": NEW_NAME}}
        )
        
        if result.modified_count > 0:
            print("Name updated successfully.")
        else:
            print("Name was already correct or update failed.")
            
        # Verify
        updated_user = users_col.find_one({"email": EMAIL})
        print(f"Verified new name in DB: {updated_user.get('name')}")
        
    else:
        print(f"User with email {EMAIL} not found.")

if __name__ == "__main__":
    update_user_name()
