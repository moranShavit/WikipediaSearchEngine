# backend_planing.md
Backend planning for a modular retrieval engine (GCP bucket now, local VM later)

## Goals
- Build a **modular backend** that powers the Flask endpoints in `search_frontend.py`.
- Support **two storage strategies**:
  1. **GCP bucket strategy (current)** – read posting lists using `inverted_index_gcp.py`.
  2. **Local VM strategy (future)** – use a different inverted index reader while keeping identical module interfaces.

## High-level architecture
The backend is composed of the following modules:

- `meta_data_module`
- `title_module`
- `body_module`
- `anchor_module`
- `engine_search`

Only `engine_search` is accessed directly by the Flask app. All other modules are internal and reusable.

---

## Storage strategy (no IndexAccessor)
Each retrieval module holds a direct reference to the index object it needs (e.g., title index, body index, anchor index).

- In **GCP mode (current)**: each module loads an `idx.InvertedIndex` using `InvertedIndex.read_index(base_dir, name, bucket_name=...)`.
- In **Local mode (future)**: each module will load a different index implementation (new file) but must expose the same minimal method used by modules: `read_a_posting_list(...)` plus access to `df` and `posting_locs`.

### Thread-safety & parallel reads (important)
Parallel queries are implemented using threads (parallel-by-term).

**Good news:** in `inverted_index_gcp.py`, `InvertedIndex.read_a_posting_list(...)` creates a *new* `MultiFileReader` inside the call (`with closing(MultiFileReader(...)) as reader:`), so each invocation has its own file handles and internal state. This means multiple threads can safely call `index.read_a_posting_list(...)` on the **same shared index object** in parallel, because the index object is read-only during serving and the per-call reader state is not shared.

**Rule:** do not share a single `MultiFileReader` instance across threads. Either:
- call `index.read_a_posting_list(...)` (simplest + safe), or
- later optimization: use a *thread-local* `MultiFileReader` per worker thread (not shared).

---

## Module: `meta_data_module`

### Purpose
Centralized access to all document-level metadata, aligned by **pos**.

### Data owned
- `doc_id_to_pos.npy` – maps `doc_id -> pos`
- Arrays indexed by `pos`:
  - `doc_norm_body.npy`
  - `inv_doc_len_body.npy`
  - `title.npy`
  - `page_rank.npy`

### API
One method per attribute:
- `get_doc_norm_body(doc_id)`
- `get_inv_doc_len_body(doc_id)`
- `get_title(doc_id)`
- `get_page_rank(doc_id)`

All retrieval modules receive an instance of this module in their constructor.

---

## Module: `title_module` (UPDATED)

### Endpoint support
Supports `/search_title`.

### Purpose
Retrieve all documents whose **title contains query terms**, ranked by the number of **distinct query terms** appearing in the title.

### Module fields
- `self.index`: the **title** `InvertedIndex` object loaded once at init
- `self.base_dir`, `self.bucket_name`
- `self.is_text_posting` (if relevant for the title index)

### Output
- `list[tuple[int, int]]`
- `(doc_id, matched_distinct_query_terms_in_title)`
- Sorted by match count descending

### Core design decisions
- **Sparse accumulation using dictionaries**
- **Parallelized by query term**
- No dense vectors or global shared state

### Parallel execution model
1. Deduplicate query terms
2. For each term (in parallel):
   - Read posting list from the title index via the shared index object:
     - `self.index.read_a_posting_list(self.base_dir, term, self.bucket_name, is_text_posting=...)`
   - Produce a local dictionary:
     ```python
     {doc_id: 1}
     ```
3. Merge all local dictionaries in the main thread by summing values per `doc_id`

### Thread-safety note
It is safe for multiple threads to call `self.index.read_a_posting_list(...)` concurrently because each call constructs its own `MultiFileReader` (no shared reader state). Do **not** share a single `MultiFileReader` instance across threads.

---

## Module: `body_module`

### Endpoint support
Supports `/search_body`.

### Purpose
Compute TF‑IDF–based similarity scores for document bodies.

### Output
- `list[tuple[int, float]]`
- `(doc_id, score)` sorted descending

### Design
- Sparse accumulation of scores per document
- Parallelized by query term
- Each worker:
  - Reads posting list
  - Computes partial scores
  - Returns a local sparse structure

### Merging strategy
- Worker outputs are merged in the main thread
- Metadata (`doc_norm_body`, `inv_doc_len_body`) is applied via `meta_data_module`

Dense vectors of corpus size are explicitly avoided.

---

## Module: `anchor_module` (UPDATED)

### Status
The `/search_anchor` endpoint ranking is not finalized yet.

### Current baseline purpose
Retrieve documents whose **anchor text contains query terms**, ranked by the number of **distinct query terms** matched.

### Module fields
- `self.index`: the **anchor** `InvertedIndex` object loaded once at init
- `self.base_dir`, `self.bucket_name`
- `self.is_text_posting` (typically `False` in your current usage)

### Output (baseline)
- `list[tuple[int, int]]`
- `(doc_id, matched_distinct_query_terms_in_anchor)`

### Core design decisions
- Same **parallel sparse dictionary approach** as `title_module`
- Parallelized by query term
- No dense vectors

### Parallel execution model
1. Deduplicate query terms
2. For each term (parallel):
   - Read anchor posting list via the shared index object:
     - `self.index.read_a_posting_list(self.base_dir, term, self.bucket_name, is_text_posting=...)`
   - Produce `{doc_id: 1}`
3. Merge dictionaries by summing counts

### Thread-safety note
Like title, it is safe for multiple threads to call `self.index.read_a_posting_list(...)` concurrently because each call creates its own `MultiFileReader`. Do **not** share a single `MultiFileReader` instance across threads.

### Future extensions
- Anchor weighting
- TF‑based anchor scoring
- Integration into learned ranking

The current structure supports all future extensions without refactoring.

---

## Module: `engine_search`

### Purpose
The main orchestrator and **single entry point** between Flask and the backend.

### Responsibilities
- Initialize storage strategy (GCP or local)
- Construct and own all modules
- Route search requests to the appropriate modules
- Combine multiple signals into final rankings

### Initialization
`EngineSearch(storage_strategy, config)`

- Creates the appropriate `IndexAccessor`
- Passes it to `title_module`, `body_module`, and `anchor_module`

### Ranking for `/search`
Initial ranking combines:
- Body score
- Title match boost
- PageRank

Weights are configurable and will later be optimized using training data.

---

## Global threading & accumulation strategy

Used consistently in:
- `title_module`
- `body_module`
- `anchor_module`

### Rules
- Parallelize by query term
- Each thread produces local sparse results
- No shared mutable state in workers
- Main thread merges results

### Rationale
- Posting reads are I/O-bound
- Posting lists are sparse relative to corpus size
- Sparse dicts scale with matched documents, not total corpus size

---

## Implementation order
1. `meta_data_module`
2. `IndexAccessor` + GCP implementation
3. `title_module`
4. `body_module`
5. `engine_search`
6. `anchor_module` extensions

---

## What this design enables
- Clean swap from GCP to local storage
- Efficient query-time memory usage
- Easy extension to advanced ranking models

