# body_module.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Iterable, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import math

import inverted_index_gcp as idx  # InvertedIndex.read_index + read_a_posting_list :contentReference[oaicite:3]{index=3}


StorageMode = Literal["gcs", "local"]


@dataclass(frozen=True)
class BodyIndexConfig:
    """
    GCS only for now (local inverted index will be implemented later).
    """
    base_dir: str          # e.g. "indexes/body"
    index_name: str        # e.g. "body_index"
    mode: StorageMode = "gcs"
    bucket_name: Optional[str] = None
    is_text_posting: bool = False


class BodyModule:
    """
    TF-IDF / cosine-style scoring for BODY.

    Output: list[(doc_id, score)] sorted descending. :contentReference[oaicite:4]{index=4}

    Parallel model: parallel-by-term, each worker reads posting list and produces
    local sparse scores; main thread merges. :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, config: BodyIndexConfig, meta_data_module):
        self.base_dir = config.base_dir
        self.index_name = config.index_name
        self.mode = config.mode
        self.bucket_name = config.bucket_name
        self.is_text_posting = config.is_text_posting

        self.meta = meta_data_module

        if self.mode == "gcs":
            if not self.bucket_name:
                raise ValueError("bucket_name is required when mode='gcs'.")
            self.index = idx.InvertedIndex.read_index(
                base_dir=self.base_dir,
                name=self.index_name,
                bucket_name=self.bucket_name,
            )
        elif self.mode == "local":
            # local body inverted index not implemented yet (you said we’ll add later)
            raise NotImplementedError("Local body inverted index not implemented yet.")
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'gcs' or 'local'.")

        # Corpus size N: use metadata mapping length (fast + already in RAM)
        # (doc_id_to_pos is loaded in MetaDataModule.__init__)
        self.N = int(self.meta.doc_id_to_pos.shape[0])

    # -------------------------
    # Public API
    # -------------------------
    def search(
        self,
        query_terms: Iterable[str],
        max_workers: int = 16,
        top_k: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Args:
            query_terms: tokenized query (no stemming, stopwords removed).
            max_workers: parallel threads (posting reads are I/O-bound).
            top_k: return up to K results (default 100 for /search_body).

        Returns:
            list of (doc_id, score) sorted descending.
        """
        q_terms = list(query_terms)
        if not q_terms:
            return []

        q_tf = Counter(q_terms)          # keep query tf
        uniq_terms = list(q_tf.keys())   # parallel by unique term

        scores = defaultdict(float)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(self._score_term_contrib, term, q_tf[term])
                for term in uniq_terms
            ]
            for fut in as_completed(futures):
                local = fut.result()
                if not local:
                    continue
                for doc_id, s in local.items():
                    scores[doc_id] += s

        if not scores:
            return []

        # Sort and cut top_k
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if top_k and top_k > 0:
            ranked = ranked[:top_k]
        return ranked

    def search_bm25(
        self,
        query_terms: Iterable[str],
        max_workers: int = 16,
        top_k: int = 100,
        *,
        k1: float = 1.8,
        b: float = 0.1,
        use_bm25plus: bool = False,
        delta: float = 1.0,
    ) -> List[Tuple[int, float]]:
        """
        BM25 scoring for BODY (no k3, query factor = 1).

        Args:
            query_terms: tokenized query (no stemming, stopwords removed).
            max_workers: parallel threads (posting reads are I/O-bound).
            top_k: return up to K results.
            k1: term frequency saturation (typical 1.2–2.0).
            b: length normalization (0–1, typical 0.75).
            use_bm25plus: enable BM25+ variant.
            delta: BM25+ delta (ignored if use_bm25plus=False).

        Returns:
            list of (doc_id, score) sorted descending.
        """
        q_terms = list(query_terms)
        if not q_terms:
            return []

        avgdl = float(self.meta.get_avg_doc_len_body())
        if avgdl <= 0.0:
            return []

        uniq_terms = set(q_terms)
        scores = defaultdict(float)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    self._bm25_term_contrib,
                    term,
                    avgdl,
                    k1,
                    b,
                    use_bm25plus,
                    delta,
                )
                for term in uniq_terms
            ]

            for fut in as_completed(futures):
                local = fut.result()
                if not local:
                    continue
                for doc_id, s in local.items():
                    scores[doc_id] += s

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if top_k and top_k > 0:
            ranked = ranked[:top_k]
        return ranked

    def _bm25_term_contrib(
        self,
        term: str,
        avgdl: float,
        k1: float,
        b: float,
        use_bm25plus: bool,
        delta: float,
    ) -> Dict[int, float]:
        """
        Per-term BM25 worker.

        score = idf * ( tf * (k1 + 1) ) /
                        ( tf + k1 * (1 - b + b * dl / avgdl) )

        BM25+:
            score = idf * ( (tf * (k1 + 1)) / denom + delta )
        """
        df = self.index.df.get(term, 0)
        if df <= 0:
            return {}

        # Standard BM25 IDF
        idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        try:
            pl = self.index.read_a_posting_list(
                self.base_dir,
                term,
                self.bucket_name,
                is_text_posting=self.is_text_posting,
            )
        except Exception:
            return {}

        if not pl:
            return {}

        out: Dict[int, float] = {}

        for doc_id, tf in pl:
            doc_id = int(doc_id)
            tf = float(tf)

            inv_len = self.meta.get_inv_doc_len_body(doc_id)
            if inv_len <= 0.0:
                continue

            dl = 1.0 / inv_len
            norm = (1.0 - b) + b * (dl / avgdl)
            denom = tf + k1 * norm
            if denom <= 0.0:
                continue

            base = (tf * (k1 + 1.0)) / denom

            if use_bm25plus:
                score = idf * (base + delta)
            else:
                score = idf * base

            out[doc_id] = out.get(doc_id, 0.0) + score

        return out


    # -------------------------
    # Internal: per-term worker
    # -------------------------
    def _score_term_contrib(self, term: str, qtf: int) -> Dict[int, float]:
        """
        Worker does:
          - read posting list for term
          - for each (doc, tf):
              tf_norm = tf * inv_doc_len_body(doc)
              idf = log( (N+1) / (df+1) )
              contrib = (qtf * idf) * (tf_norm * idf) / doc_norm_body(doc)

        We skip dividing by query_norm on purpose (constant per query, ranking unchanged).
        """
        # term not in index => no contribution
        df = self.index.df.get(term, 0)
        if df <= 0:
            return {}

        # IDF (smoothed)
        idf = math.log((self.N + 1.0) / (df + 1.0))

        try:
            pl = self.index.read_a_posting_list(
                self.base_dir,
                term,
                self.bucket_name,
                is_text_posting=self.is_text_posting,
            )
        except Exception:
            return {}

        if not pl:
            return {}

        # query weight (can take idf in corpus incount also for query)
        # qw = float(qtf) * idf
        qw = float(qtf)

        out: Dict[int, float] = {}

        for doc_id, tf in pl:
            doc_id = int(doc_id)

            inv_len = self.meta.get_inv_doc_len_body(doc_id)
            if inv_len == 0.0:
                continue

            doc_norm = self.meta.get_doc_norm_body(doc_id)
            if doc_norm == 0.0:
                continue

            # normalized TF in doc
            tf_norm = float(tf) * inv_len

            # document weight for this term
            dw = tf_norm * idf

            # cosine-style contribution (query_norm omitted)
            out[doc_id] = out.get(doc_id, 0.0) + (qw * dw) / doc_norm

        return out
