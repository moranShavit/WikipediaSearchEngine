# Preprocessing Plan – Search Engine Backend

This document defines **all offline preprocessing tasks** required to support the Flask search backend. These tasks are executed on a **Spark cluster** and write their outputs to **Google Cloud Storage (GCS)**. The Flask service assumes all outputs described here already exist and are immutable.

The goal of this file is to:
- Keep preprocessing requirements explicit and well-scoped
- Allow incremental implementation and validation
- Serve as a fast context reset when resuming work

---

## Global Constraints & Assumptions

- Tokenization and stopword removal **must exactly match** the logic used during index construction
- All indexes are **disk-based** and written using `inverted_index_gcp.py`
- Posting lists are **not loaded fully into memory** at query time
- Each preprocessing job is independent and can be rerun safely

---

## Preprocessing Tasks Overview

| ID | Task Name | Purpose | Output Location |
|----|----------|--------|----------------|
| P1 | Body Inverted Index | Core retrieval over document body | `indexes/body/` | finish
| P2 | Title Inverted Index | Title-based retrieval | `indexes/title/` | finish
| P3 | Anchor Inverted Index | Anchor-text-based retrieval | `indexes/anchor_text/` | finish
| P4 | Body TF-IDF Norms | Cosine similarity normalization | `metadata/doc_norm_body.pkl` | finish
| P5 | DocID → Title Map | Result formatting | `metadata/docid_to_title.pkl` | finish
| P6 | PageRank Processing | Ranking signal | `metadata/pagerank.pkl` | finish
| P7 | PageViews (Aug 2021) | Popularity signal | `metadata/pageviews_2021_08.pkl` |

---

## P1 – Body Inverted Index

**Purpose**  
Support `/search_body` and future `/search` endpoints.

**Input**  
- `(doc_id, body_text)`

**Processing**  
- Tokenize using provided regex
- Remove stopwords
- Count term frequency per document
- Group by term
- Sort posting lists by `doc_id`
- Partition by `token2bucket_id`

**Output**  
- `indexes/body/body_index.pkl` (metadata)
- `indexes/body/body_index_*.bin` (posting shards)

**Notes**  
- Already partially implemented
- Serves as the canonical tokenizer reference

---

## P2 – Title Inverted Index

**Purpose**  
Support `/search_title` endpoint (distinct query word matches).

**Input**  
- `(doc_id, title_text)`

**Processing**  
- Tokenize titles
- Remove stopwords
- Count term frequency per document (tf stored but not required for ranking)
- Build inverted index using same infrastructure as body

**Output**  
- `indexes/title/title_index.pkl`
- `indexes/title/title_index_*.bin`

**Query-time usage**  
- For each query token, retrieve posting list
- Count number of **distinct query tokens** matched per document

---

## P3 – Anchor Inverted Index

**Purpose**  
Support `GET /search_anchor` by enabling fast retrieval of documents whose **anchor text** matches query terms.

**Key requirement**  
Build a **disk-based inverted index** (same storage pattern as the body index) where each term maps to a posting list of documents associated with that term in anchor text.

**Data structure**  
- Inverted index: `term → posting list`
- Posting entry format: `(doc_id, tf)`

**Output**  
- `indexes/anchor/anchor_index.pkl`
- `indexes/anchor/anchor_index_*.bin`

**Notes**  
- This task focuses on the index artifact definition and storage format.
- Specific details about raw input schema and Spark transformations will be defined when implementing the Spark jobs.

---



## P4 – Body TF-IDF Document Norms

**Purpose**  
Enable efficient cosine similarity computation for `/search_body`.

**Input**  
- Body inverted index metadata (`df`)
- Body posting lists

**Processing**  
- Compute `idf = log(N / df)`
- For each document, accumulate:
  ```
  sum((tf * idf)^2)
  ```
- Take square root per document

**Output**  
- `metadata/doc_norm_body.pkl`  
  Mapping: `doc_id → norm`
- `metadata/N.pkl` (optional, can be embedded)

---

## P5 – DocID → Title Mapping (Memory-Mapped Binary)

**Purpose**  
Convert internal document IDs into human-readable titles at query time without large memory overhead.

This task provides **forward metadata** (`doc_id → title`). It is **not** used for retrieval, only for formatting search results.

---

### Storage Strategy (Chosen)

**Memory-mapped binary file (read-only)**

Rationale:
- ~6 million documents
- Lookup by `doc_id` only (no search)
- Read-only at runtime
- Must be fast and memory-efficient

A Python dictionary or pickle would consume multiple GBs of RAM. Instead, titles are stored in a compact binary format and accessed via OS-level memory mapping (`mmap`).

---

### File Format

Two binary components are written during preprocessing:

1. **Offsets array** (`titles_offsets.bin`)
   - `offsets[i]` = starting byte position of title for `doc_id = i`
   - `offsets[i+1] - offsets[i]` = title length in bytes

2. **Title blob** (`titles_data.bin`)
   - All titles concatenated as UTF-8 bytes

Logical layout:
```
[titles_offsets.bin]
[ titles_data.bin ]
```

---

### Runtime Access Pattern

At Flask startup:
- Open both files
- Memory-map them in read-only mode

At query time (for each result doc_id):
```python
start = offsets[doc_id]
end   = offsets[doc_id + 1]
title = data[start:end].decode('utf-8')
```

Only the required bytes are paged into memory by the OS.

---

### Output

- `metadata/titles_offsets.bin`
- `metadata/titles_data.bin`

These files replace `docid_to_title.pkl` and are treated as immutable artifacts.

---



## P6 – PageRank Processing

**Purpose**  
Support `/get_pagerank` and future ranking logic.

**Input**  
- Raw PageRank output (CSV / Parquet)

**Processing**  
- Convert to mapping: `doc_id → pagerank`
- Normalize if required

**Output**  
- `metadata/pagerank.pkl`

---

## P7 – PageViews (August 2021)

**Purpose**  
Support `/get_pageview` endpoint.

**Input**  
- Wikimedia pageviews logs

**Processing**  
- Filter August 2021
- Aggregate views per document

**Output**  
- `metadata/pageviews_2021_08.pkl`

---

## Validation Checklist (Per Task)

For each preprocessing job:
- [ ] Output files exist in GCS
- [ ] Index metadata loads without error
- [ ] Sample posting lists decode correctly
- [ ] Flask endpoint returns expected format

---

## Implementation Order (Recommended)

1. P5 – DocID → Title map
2. P6 – PageRank
3. P7 – PageViews
4. P2 – Title index
5. P3 – Anchor index
6. P4 – TF-IDF norms
7. P1 – Rebuild body index if needed

---

## Notes for Future `/search` Endpoint

The main `/search` endpoint will likely combine:
- Body cosine similarity
- Title / anchor matches
- PageRank
- PageViews

Preprocessing outputs in this document are designed to support this without further offline computation.

