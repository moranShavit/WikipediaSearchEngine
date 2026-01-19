# Wikipedia Search Engine

A scalable, production-ready search engine for Wikipedia articles built with Python, Flask, and Google Cloud Platform. This project implements multiple retrieval methods including TF-IDF, BM25, and PageRank-based ranking, with efficient storage and retrieval using Google Cloud Storage.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Technologies](#technologies)

## 🎯 Overview

This search engine provides fast and accurate search capabilities over Wikipedia articles. It separates offline preprocessing (using Apache Spark) from online serving (using Flask), enabling efficient scaling and memory management. All indexes and metadata are stored in Google Cloud Storage, allowing the service to load only necessary data on-demand.

### Key Design Principles

- **Offline Preprocessing**: Heavy computation (index building, statistics calculation) is done offline using Spark
- **Online Efficiency**: The Flask service loads lightweight metadata and reads posting lists from GCS on-demand
- **Scalability**: Indexes are stored in GCS, enabling horizontal scaling and efficient memory usage
- **Multiple Retrieval Methods**: Supports body, title, and anchor text search with different ranking strategies

## 🏗️ Architecture

```
┌─────────────────────┐
│  Offline Processing │
│   (Apache Spark)    │
│  - Build indexes    │
│  - Calculate stats  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Google Cloud Storage│
│   (GCS Bucket)      │
│  - Indexes          │
│  - Metadata         │
│  - PageRank/Views   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Flask REST API     │
│   (GCP VM)          │
│  - Load metadata    │
│  - Read postings    │
│  - Rank results     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Clients           │
│  (Browser/API)      │
└─────────────────────┘
```

## ✨ Features

### Search Capabilities

1. **Full-Text Search** (`/search`)
   - Combines multiple signals (body, title, anchor, PageRank, pageviews)
   - Returns top 100 results ranked by relevance

2. **Body Search** (`/search_body`)
   - TF-IDF + cosine similarity over article bodies
   - Uses BM25 scoring for improved relevance

3. **Title Search** (`/search_title`)
   - Returns all articles with query words in title
   - Ranked by number of distinct query words matched

4. **Anchor Text Search** (`/search_anchor`)
   - Searches anchor text linking to articles
   - Ranked by number of distinct query words in anchor text

### Metadata Services

- **PageRank Lookup** (`/get_pagerank`): Returns PageRank scores for given article IDs
- **Pageview Lookup** (`/get_pageview`): Returns August 2021 pageview counts for given article IDs

## 📁 Project Structure

```
WikipediaSearchEngine/
├── src/
│   ├── search_engine.py          # Main search engine implementation
│   ├── search_frontend.py        # Flask REST API endpoints
│   ├── body_module.py            # Body text indexing and retrieval
│   ├── title_module.py           # Title indexing and retrieval
│   ├── anchor_module.py          # Anchor text indexing and retrieval
│   ├── meta_data_module.py       # Metadata management (PageRank, pageviews, titles)
│   ├── inverted_index_gcp.py     # GCS-backed inverted index implementation
│   ├── modules_checks/           # Testing and validation scripts
│   └── spark_notebooks/          # Spark preprocessing notebooks
├── planing/                      # Project planning documents
│   ├── project_context.md
│   ├── preprocessing_plan.md
│   └── backend_planing.md
├── queries_train.json            # Training queries for evaluation
├── requirements.txt              # Python dependencies
├── startup_script_gcp.sh        # GCP VM startup script
└── run_frontend_in_gcp.sh       # Frontend deployment script
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Google Cloud Platform account with:
  - GCS bucket for storing indexes
  - GCP VM instance (for deployment)
- Apache Spark (for offline preprocessing)

### Local Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK stopwords** (if not already downloaded):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. **Configure GCS bucket**:
   - Update the bucket name in `src/search_frontend.py` (line 11)
   - Ensure your GCP credentials are configured:
     ```bash
     gcloud auth application-default login
     ```

## 💻 Usage

### Running Locally

1. **Set up the search engine** (ensure indexes are in GCS bucket):
   ```bash
   cd src
   python search_frontend.py
   ```

2. **Test the API**:
   ```bash
   # Search endpoint
   curl "http://localhost:8080/search?query=artificial+intelligence"
   
   # PageRank lookup
   curl -X POST http://localhost:8080/get_pagerank \
     -H "Content-Type: application/json" \
     -d '[3434750, 10568, 32927]'
   ```

### Example Python Client

```python
import requests

# Base URL (update with your server domain)
BASE_URL = "http://localhost:8080"

# Search query
response = requests.get(f"{BASE_URL}/search", params={"query": "machine learning"})
results = response.json()
print(f"Found {len(results)} results")

# Get PageRank for specific articles
wiki_ids = [3434750, 10568, 32927]
response = requests.post(f"{BASE_URL}/get_pagerank", json=wiki_ids)
pageranks = response.json()
print(f"PageRank values: {pageranks}")
```

## 🔌 API Endpoints

### GET `/search`
Main search endpoint combining multiple signals.

**Parameters:**
- `query` (string): Search query

**Returns:**
- List of tuples `(wiki_id, title)` - up to 100 results

**Example:**
```
GET /search?query=python+programming
```

### GET `/search_body`
Search using TF-IDF and cosine similarity over article bodies.

**Parameters:**
- `query` (string): Search query

**Returns:**
- List of tuples `(wiki_id, title)` - up to 100 results

### GET `/search_title`
Search articles with query words in title.

**Parameters:**
- `query` (string): Search query

**Returns:**
- List of tuples `(wiki_id, title)` - ALL matching results

### GET `/search_anchor`
Search using anchor text linking to articles.

**Parameters:**
- `query` (string): Search query

**Returns:**
- List of tuples `(wiki_id, title)` - ALL matching results

### POST `/get_pagerank`
Get PageRank scores for article IDs.

**Request Body:**
- JSON array of integers: `[1, 5, 8]`

**Returns:**
- JSON array of floats: `[0.0, 0.0, 0.0]`

**Example:**
```python
import requests
requests.post('http://YOUR_SERVER/get_pagerank', json=[3434750, 10568])
```

### POST `/get_pageview`
Get August 2021 pageview counts for article IDs.

**Request Body:**
- JSON array of integers: `[1, 5, 8]`

**Returns:**
- JSON array of integers: `[100, 200, 300]`

**Example:**
```python
import requests
requests.post('http://YOUR_SERVER/get_pageview', json=[15580374, 1610886])
```

## ☁️ Deployment

### GCP Deployment

1. **Create a GCP VM instance** with appropriate resources

2. **Upload project files** to the VM:
   ```bash
   gcloud compute scp --recurse . VM_NAME:/home/USER/
   ```

3. **Run the startup script** (or manually set up):
   ```bash
   bash startup_script_gcp.sh
   ```

4. **Start the Flask service**:
   ```bash
   cd src
   python search_frontend.py
   ```

5. **Configure firewall** to allow traffic on port 8080:
   ```bash
   gcloud compute firewall-rules create allow-search-engine \
     --allow tcp:8080 \
     --source-ranges 0.0.0.0/0
   ```

### Using the Deployment Script

```bash
bash run_frontend_in_gcp.sh
```

## 🛠️ Technologies

- **Python 3.10+**: Core language
- **Flask**: REST API framework
- **Apache Spark**: Offline preprocessing and index building
- **Google Cloud Storage**: Index and metadata storage
- **NLTK**: Natural language processing (tokenization, stopwords)
- **NumPy**: Numerical computations
- **Google Cloud Platform**: Infrastructure and deployment

## 📊 Index Structure

The search engine uses a disk-based inverted index design:

- **Index Metadata** (`*.pkl`): Lightweight metadata loaded at startup
  - Document frequencies
  - Posting list locations
  - Term statistics

- **Posting Lists** (`*.bin`): Binary files stored in GCS
  - Split across multiple shard files
  - Lazy-loaded on-demand
  - Efficient memory usage

## 📝 Notes

- The search engine uses the staff-provided tokenizer from Assignment 3
- Stopwords are removed using NLTK's English stopwords plus corpus-specific stopwords
- PageRank and pageview data are from Wikipedia's August 2021 snapshot
- All indexes are immutable once written to GCS

## 📄 License

This project is part of a university course assignment.

## 👥 Authors

Moran Shavit & Tamar Hagbiy

---

For detailed implementation notes and planning documents, see the `planing/` directory.

