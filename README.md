# DataGatorLite 🐊

🔗 Try it now: [DataGator Lite](https://datagator-lite.onrender.com/)

**DataGatorLite** is a lightweight web app built with Flask for uploading datasets and performing fuzzy matching against standardized International Futures territory names. It supports Excel and CSV formats, and allows manual review and editing of suggestions before exporting the cleaned data to a downloadable CSV file.

---

## 🚀 Features

- Upload Excel or CSV files
- Smart detection of header rows
- Select specific rows/columns for name matching
- Fuzzy matching using RapidFuzz with a threshold control
- Manual override and review interface
- Export the cleaned dataset
- Secure session handling (server-side)

---

## 🛠️ Getting Started (Local)

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/datagator_flask_mongo.git
cd datagator_flask_mongo
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4.a Create a `.env` file
```bash
cp .env.example .env
```
Fill in the required fields (`DB_URI_PART1`, etc.)
Use this route to directly connect to a MongoDB database with territory mapping

### 4.b Connect to JSON data locally

Use the JSON file under the data folder to read the territory mapping database

### 5. Run the app
```bash
python app.py
```

App will be available via a local host link.

---

## 📄 License

MIT License — feel free to use and adapt.

---

## 🙌 Credits

Built by Yutang Xiong  
Contact: [yutang.xiong@du.edu](mailto:yutang.xiong@du.edu)
