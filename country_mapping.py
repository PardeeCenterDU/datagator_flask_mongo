from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
part1 = os.getenv("DB_URI_PART1")
part2 = os.getenv("DB_URI_PART2")
part3 = os.getenv("DB_URI_PART3")
part4 = os.getenv("DB_URI_PART4")
# Connect to MongoDB
url = f"mongodb+srv://{part1}:{part2}@{part3}/?retryWrites=true&w=majority"
client = MongoClient(url)
db = client[part4]  # Replace with your actual DB name
countries_collection = db['names_mappings']  # Replace with your actual collection name


if __name__ == "__main__":
    input_country = "Untied States of America"
    alter_country = countries_collection.find(input_country)
    for i in alter_country:
        print(i)