# title_module.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Iterable, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import inverted_index_gcp as idx  # uses InvertedIndex.read_index + read_a_posting_list :contentReference[oaicite:1]{index=1}


StorageMode = Literal["gcs", "local"]


@dataclass(frozen=True)
class AnchorIndexConfig:
    """
    Keeps the module flexible:
      - GCS mode: bucket_name must be provided
      - local mode: bucket_name ignored (must be None / omitted)
    """
    base_dir: str         # e.g. "indexes/title"
    index_name: str       # e.g. "title_index"
    mode: StorageMode = "gcs"
    bucket_name: Optional[str] = None
    is_text_posting: bool = False  # keep for compatibility with the index reader signature


class AnchorModule:
    """
    Retrieve documents whose TITLE contains query terms, ranked by
    the number of DISTINCT query terms matched in the title.

    Output:
      list[(doc_id, matched_distinct_query_terms)] sorted descending.

    Thread-safety:
      Safe to call index.read_a_posting_list concurrently because each call
      creates its own MultiFileReader inside the method. :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, config: AnchorIndexConfig):
        self.base_dir = config.base_dir
        self.index_name = config.index_name
        self.mode = config.mode
        self.bucket_name = config.bucket_name
        self.is_text_posting = config.is_text_posting

        if self.mode == "gcs":
            if not self.bucket_name:
                raise ValueError("bucket_name is required when mode='gcs'.")
            self.index = idx.InvertedIndex.read_index(
                base_dir=self.base_dir,
                name=self.index_name,
                bucket_name=self.bucket_name,
            )
        elif self.mode == "local":
            pass


    # -------------------------
    # Public API
    # -------------------------
    def search(self, query_terms: Iterable[str], max_workers: int = 16) -> List[Tuple[int, int]]:
        """
        Args:
            query_terms: already-tokenized terms (stopwords removed, no stemming),
                         as required by /search_title endpoint.
            max_workers: threads for parallel-by-term posting reads.

        Returns:
            List of (doc_id, matched_distinct_terms) sorted by:
              - matched_distinct_terms descending
              - doc_id ascending (stable tie-break)
        """
        terms = self._dedupe_terms(query_terms)
        if not terms:
            return []

        # Merge (main thread): sum values per doc_id => distinct-term match count
        scores = defaultdict(int)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self._term_to_docs_dict, term) for term in terms]
            for fut in as_completed(futures):
                doc_ids = fut.result()
                for doc_id, tf in doc_ids:
                    scores[doc_id] += tf

        # Sort by match count desc, tie-break doc_id asc
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    # -------------------------
    # Internal helpers
    # -------------------------
    def _dedupe_terms(self, query_terms: Iterable[str]) -> List[str]:
        # Preserve original order (deterministic), but ensure distinct terms
        seen = set()
        out: List[str] = []
        for t in query_terms:
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _term_to_docs_dict(self, term: str) -> List[int]:
        """
        Worker function:
          - reads posting list for one term
          - returns {doc_id: 1} for all docs containing the term in title
        """
        try:
            pl = self.index.read_a_posting_list(
                self.base_dir,
                term,
                self.bucket_name if self.mode == "gcs" else None,
                is_text_posting=self.is_text_posting,
            )
        except Exception:
            # Fail-soft: if one term has an issue, don't kill the whole query
            return []

        if not pl:
            return []

        # posting list is [(doc_id, tf)]
        # return [(doc_id, tf) for doc_id, tf in pl]
        return pl
