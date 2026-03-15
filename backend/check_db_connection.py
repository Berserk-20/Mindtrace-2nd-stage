import os
import pymongo
from pymongo import MongoClient
import sys

# Try default localhost
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "mindtrace_db"

print(f"Testing connection to: {MONGO_URI}")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print("MongoDB Connection Successful!")
    
    db = client[DB_NAME]
    collections = db.list_collection_names()
    print(f"Collections found: {collections}")
    
except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"MongoDB Connection Check Failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
