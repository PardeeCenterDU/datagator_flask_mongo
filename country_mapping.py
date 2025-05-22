import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB connection parts from .env
part1 = os.getenv("DB_URI_PART1")
part2 = os.getenv("DB_URI_PART2")
part3 = os.getenv("DB_URI_PART3")
part4 = os.getenv("DB_URI_PART4")  # should be your DB name (e.g., ifs_country_concordance)

# Mongo URI
MONGO_URI = f"mongodb+srv://{part1}:{part2}@{part3}/?retryWrites=true&w=majority"

# Folder and file for JSON export
JSON_PATH = os.path.join("data", "country_data.json")

# ----------------------------------------
# Mongo: connect and return collection
# ----------------------------------------
def get_countries_collection():
    client = MongoClient(MONGO_URI)
    db = client[part4]
    return db["names_mappings"]

# ----------------------------------------
# Export MongoDB collection to JSON
# ----------------------------------------
def export_to_json():
    countries_collection = get_countries_collection()
    docs = list(countries_collection.find({}, {"_id": 0}))  # exclude Mongo _id
    os.makedirs("data", exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"✅ Exported {len(docs)} documents to {JSON_PATH}")

# ----------------------------------------
# Load from local JSON file
# ----------------------------------------
def load_country_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
