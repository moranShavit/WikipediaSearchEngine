# Search Engine Backend – Project Overview

## Overview

This project implements the **backend of a search engine**. The backend is exposed as a **REST API using Flask** and is designed to run on a **Google Cloud Platform (GCP) VM**. All heavy data (indexes, statistics, metadata) is **preprocessed offline using Spark** and stored in a **Google Cloud Storage (GCS) bucket**. The Flask service loads lightweight metadata into memory and reads posting lists from the bucket on demand.

The project is structured to clearly separate:

- **Offline preprocessing (Spark jobs)** – build indexes and statistics
- **Online serving (Flask app)** – answer search and ranking requests

---

## High-Level Architecture

```
Offline (Spark Cluster)
    ↓  preprocess & write
Google Cloud Storage (Bucket)
    ↓  load & read
Online (Flask API on GCP VM)
    ↓  HTTP
Clients (browser / evaluation scripts / other services)
```

---

## Existing Python Files

### `search_frontend.py`

**Role:** Flask REST API (online serving layer)

This file defines the web server and all HTTP endpoints required for the search engine. It does **not** perform heavy preprocessing. Instead, it:

- Loads indexes and metadata from the GCS bucket at startup
- Handles incoming HTTP requests
- Runs retrieval and ranking logic
- Returns results as JSON

Implemented endpoints:

- `GET /search` – main search endpoint (to be implemented later)
- `GET /search_body` – TF-IDF + cosine similarity over document bodies
- `GET /search_title` – title-based retrieval (distinct query-word matches)
- `GET /search_anchor` – anchor-text-based retrieval
- `POST /get_pagerank` – PageRank lookup by document IDs
- `POST /get_pageview` – August 2021 pageview lookup by document IDs

This file is the **entry point** of the backend service when deployed.

---

### `inverted_index_gcp.py`

**Role:** Index storage and access utilities (shared by preprocessing and serving)

This file implements:

- `InvertedIndex` – an inverted index abstraction
- Writing posting lists to disk / GCS using fixed-size binary files
- Reading posting lists from GCS on demand

Key ideas:

- Posting lists are split across multiple `.bin` files
- Index metadata (df, term statistics, posting locations) is stored as a pickle file
- Designed to support large indexes that cannot fully fit in memory

This module is used:

- During **Spark preprocessing** to write indexes to the bucket
- During **Flask runtime** to read posting lists efficiently

---

## Data Currently Available in the GCS Bucket

At this stage, the bucket already contains data written using a **disk-based inverted index design**. This design deliberately separates **index metadata** from **posting lists** to enable scalable and memory-efficient retrieval.

---

### Body Inverted Index (Disk-based, GCS-backed)

The body index is **not a single file**. It consists of two layers:

#### 1. Global Index Metadata (`index.pkl`)

This pickle file stores **only metadata**, not posting lists themselves. It is loaded **once at server startup**.

Contents:

- `` – document frequency per term (`term → number of documents containing the term`)
- `` – total collection frequency per term
- `` – mapping:
  ```
  term → [(file_name, byte_offset), ...]
  ```
  which specifies exactly where each term’s posting list is stored on disk

This metadata enables:

- fast lookup of posting locations
- correct decoding of posting lists
- computation of TF-IDF / BM25 statistics

---

#### 2. Binary Posting List Files (`*.bin`)

Posting lists themselves are stored in **multiple binary shard files** in GCS, for example:

```
postings_gcp/
├── 0_000.bin
├── 0_001.bin
├── 1_000.bin
├── ...
```

Each binary file contains many posting lists written sequentially. Posting lists are split across files when a size limit is reached.

Each posting entry is encoded as:

```
(doc_id, term_frequency)
```

The index metadata (`posting_locs`) tells the system:

- which `.bin` file(s) to read
- the byte offset where the posting list begins
- how many postings to decode (via `df`)

This design allows **lazy loading** of postings directly from GCS without keeping them all in memory.

---

### How Posting Lists Are Read at Runtime

At runtime (inside the Flask server):

1. **Load index metadata once**:

   ```python
   index = InvertedIndex.read_index(
       base_dir="postings_gcp",
       name="index",
       bucket_name="ir_3_207472234"
   )
   ```

2. **Read a posting list for a query term**:

   ```python
   posting_list = index.read_a_posting_list(
       base_dir="postings_gcp",
       w="example_term",
       bucket_name="ir_3_207472234"
   )
   ```

3. **Returned format**:

   ```
   [(doc_id, tf), (doc_id, tf), ...]
   ```

Only the posting lists required for the current query are fetched from GCS, keeping memory usage low and startup fast.

---

### PageRank Scores

- Mapping: `doc_id → pagerank score`
- Stored separately as a pickle or array
- Used by `POST /get_pagerank`
- Will later be integrated into `GET /search` ranking

---

## Expected Bucket Structure (Current + Planned) (Current + Planned)

```
gs://<bucket-name>/
  indexes/
    body/
      body_index.pkl
      body_index_000.bin
      body_index_001.bin
      ...
    title/            (to be built)
    anchor/           (to be built)
  metadata/  (to be built)
    pagerank.csv currently in: pr/part-00000-01ae429d-6dc4-4410-9263-84d031c009d4-c000.csv.gz
    docid_to_title.pkl        (to be built)
    pageviews_2021_08.pkl     (to be built)
    doc_norm_body.pkl         (to be built)
```



---

## Design Principles

- **Heavy computation is offline** (Spark)
- **Online service is fast and memory-aware**
- **Indexes are immutable** once written
- **GCS is the single source of truth** for all data
- Flask server can be restarted or scaled without recomputation

---

## Next Step

The next document will define the **preprocessing tasks** in detail:

- What Spark jobs are required
- What each job produces
- What data structures are written to the bucket

This will be tracked in a dedicated preprocessing planning file.

