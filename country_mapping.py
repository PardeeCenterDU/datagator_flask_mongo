from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read from environment
part1 = os.getenv("DB_URI_PART1")
part2 = os.getenv("DB_URI_PART2")
part3 = os.getenv("DB_URI_PART3")
part4 = os.getenv("DB_URI_PART4")  # optionally rename to DB_NAME for clarity

# Construct Mongo URI
MONGO_URI = f"mongodb+srv://{part1}:{part2}@{part3}/?retryWrites=true&w=majority"

# Fork-safe function to get the collection
def get_countries_collection():
    client = MongoClient(MONGO_URI)
    db = client[part4]  # This now correctly uses your environment-defined DB name
    return db["names_mappings"]

# Optional: CLI test mode
if __name__ == "__main__":
    countries_collection = get_countries_collection()

    input_country = "Untied States of America"
    results = countries_collection.find({
        "$or": [
            {"ifs_name": input_country},
            {"alternative_names": input_country}
        ]
    })

    for doc in results:
        print(doc)
