# meta_data_module.py

from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Optional, Literal
from io import BytesIO
import os
import gzip
import csv
import numpy as np
from google.cloud import storage
import pickle
from collections import Counter
from typing import Mapping


# ============
# GCS helpers 
# ============
PROJECT_ID = "informationretrival-480208"

def get_bucket(bucket_name: str):
    return storage.Client(PROJECT_ID).bucket(bucket_name)

def _download_blob_bytes(bucket_name: str, blob_path: str) -> bytes:
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: gs://{bucket_name}/{blob_path}")
    return blob.download_as_bytes()


# =========================
# Paths config
# =========================

@dataclass(frozen=True)
class MetaDataPaths:
    """
    In GCS mode: these are blob paths within the bucket.
    In LOCAL mode: these are filesystem paths (absolute or relative).
    """
    # numpy arrays
    doc_id_to_pos: str        # <-- FILL THIS PATH
    doc_norm_body: str        # <-- FILL THIS PATH
    inv_doc_len_body: str     # <-- FILL THIS PATH

    # titles binary files
    titles_data: str          # <-- FILL THIS PATH
    titles_offsets: str       # <-- FILL THIS PATH

    pagerank_csv_gz: Optional[str] = None  # <-- FILL THIS PATH
    pageviews_pkl: Optional[str] = None


StorageMode = Literal["gcs", "local"]


class MetaDataModule:
    """
    Loads metadata artifacts into RAM at init, then serves constant-time lookups.

    Modes:
      - mode="gcs": read from Google Cloud Storage bucket via download_as_bytes
      - mode="local": read from local filesystem (dev)

    NOTE: Both modes load into the same fields:
      self.doc_id_to_pos, self.doc_norm_body, self.inv_doc_len_body,
      self.titles_offsets, self.titles_data
    """

    def __init__(
        self,
        paths: MetaDataPaths,
        mode: StorageMode = "gcs",
        bucket_name: Optional[str] = None,
    ):
        """
        Args:
            paths: MetaDataPaths pointing to artifacts (GCS blob paths or local paths).
            mode: "gcs" or "local".
            bucket_name: required if mode="gcs". Ignored if mode="local".
        """
        self.paths = paths
        self.mode = mode
        self.bucket_name = bucket_name

        if self.mode == "gcs":
            if not bucket_name:
                raise ValueError("bucket_name is required when mode='gcs'.")

            # ---- Load numpy arrays into RAM (from GCS) ----
            self.doc_id_to_pos = self._load_npy_gcs(paths.doc_id_to_pos)
            self.doc_norm_body = self._load_npy_gcs(paths.doc_norm_body)
            self.inv_doc_len_body = self._load_npy_gcs(paths.inv_doc_len_body)

            # ---- Load titles bins into RAM (from GCS) ----
            offsets_bytes = _download_blob_bytes(bucket_name, paths.titles_offsets)
            data_bytes = _download_blob_bytes(bucket_name, paths.titles_data)

            pr_bytes = _download_blob_bytes(bucket_name, paths.pagerank_csv_gz)

            if paths.pageviews_pkl:
                pv_bytes = _download_blob_bytes(bucket_name, paths.pageviews_pkl)


        elif self.mode == "local":
            # ---- Load numpy arrays into RAM (from local) ----
            self.doc_id_to_pos = self._load_npy_local(paths.doc_id_to_pos)
            self.doc_norm_body = self._load_npy_local(paths.doc_norm_body)
            self.inv_doc_len_body = self._load_npy_local(paths.inv_doc_len_body)

            # ---- Load titles bins into RAM (from local) ----
            offsets_bytes = self._read_file_bytes(paths.titles_offsets)
            data_bytes = self._read_file_bytes(paths.titles_data)

            pr_bytes = self._read_file_bytes(paths.pagerank_csv_gz)

            if paths.pageviews_pkl:
                pv_bytes = self._read_file_bytes(paths.pageviews_pkl)

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'gcs' or 'local'.")
        


        # Determine invalid position sentinel
        if self.doc_id_to_pos.dtype == np.uint32:
            self.INVALID_POS = np.uint32(2**32 - 1)
        else:
            self.INVALID_POS = -1

        # Parse titles blobs
        self.titles_offsets = np.frombuffer(offsets_bytes, dtype=np.uint64)
        self.titles_data = np.frombuffer(data_bytes, dtype=np.uint8)

        if self.titles_offsets.size < 2:
            raise ValueError("titles_offsets is too small (need at least 2 offsets).")

        # build page rank array by pos
        self.pagerank_by_pos = self._build_pagerank_by_pos(pr_bytes)

        if pv_bytes is not None:
            wid2pv = pickle.loads(pv_bytes)  # Counter or dict

            if not isinstance(wid2pv, (dict, Counter)):
                raise TypeError(f"pageviews_pkl must unpickle to dict/Counter, got {type(wid2pv)}")

            self.pageviews_by_pos = self._build_pageviews_by_pos(wid2pv)

            # free the huge dict from RAM
            del wid2pv

        self.avg_doc_len_body = self._compute_avg_doc_len_body()

    # -------------------------
    # Load helpers (GCS/local)
    # -------------------------
    def _load_npy_gcs(self, blob_path: str) -> np.ndarray:
        b = _download_blob_bytes(self.bucket_name, blob_path)  # type: ignore[arg-type]
        return np.load(BytesIO(b), allow_pickle=False)

    def _load_npy_local(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file not found: {file_path}")
        return np.load(file_path, allow_pickle=False)

    def _read_file_bytes(self, file_path: str) -> bytes:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file not found: {file_path}")
        with open(file_path, "rb") as f:
            return f.read()
        
    # def _build_pagerank_by_pos(self, csv_gz_bytes: bytes) -> np.ndarray:
    #     """
    #     Expects a gzipped CSV with (doc_id, pagerank) per row.
    #     Handles:
    #       - with/without header
    #       - extra columns (uses first int-ish as doc_id, first float-ish as rank)
    #     Produces an array pagerank_by_pos aligned to pos.
    #     """
    #     # pos space size: same as doc_norm_body (indexed by pos)
    #     pr_by_pos = np.zeros(self.doc_norm_body.shape[0], dtype=np.float32)

    #     with gzip.GzipFile(fileobj=BytesIO(csv_gz_bytes), mode="rb") as gz:
    #         # decode as text for csv reader
    #         text = gz.read().decode("utf-8", errors="replace").splitlines()

    #     reader = csv.reader(text)
    #     rows = list(reader)
    #     if not rows:
    #         return pr_by_pos

    #     # Detect header (first row not parseable as numbers)
    #     start_i = 0
    #     if rows:
    #         r0 = rows[0]
    #         try:
    #             _ = int(r0[0])
    #             _ = float(r0[1])
    #         except Exception:
    #             start_i = 1  # skip header

    #     for r in rows[start_i:]:
    #         if not r:
    #             continue

    #         doc_id = None
    #         rank = None

    #         # find first int-like field
    #         for x in r:
    #             try:
    #                 doc_id = int(x)
    #                 break
    #             except Exception:
    #                 continue

    #         # find first float-like field (after doc_id is fine, but we’ll just scan)
    #         for x in r:
    #             try:
    #                 rank = float(x)
    #                 break
    #             except Exception:
    #                 continue

    #         if doc_id is None or rank is None:
    #             continue

    #         pos = self._doc_id_to_pos(doc_id)
    #         if pos is None:
    #             continue

    #         pr_by_pos[pos] = rank

    #     return pr_by_pos

    def _build_pagerank_by_pos(self, csv_gz_bytes: bytes) -> np.ndarray:
        """
        File format (confirmed):
        doc_id,pagerank
        3434750,9913.728782160782
        ...
        No header, exactly 2 columns.

        Builds pagerank_by_pos aligned to pos (float32 for memory).
        Missing docs remain 0.0.
        """
        pr_by_pos = np.zeros(self.doc_norm_body.shape[0], dtype=np.float32)

        with gzip.GzipFile(fileobj=io.BytesIO(csv_gz_bytes), mode="rb") as gz:
            text_stream = io.TextIOWrapper(gz, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(text_stream)

            for row in reader:
                # Safe guards
                if not row or len(row) < 2:
                    continue

                try:
                    doc_id = int(row[0])
                    rank = float(row[1])
                except Exception:
                    continue

                pos = self._doc_id_to_pos(doc_id)
                if pos is None:
                    continue

                pr_by_pos[pos] = rank

        return pr_by_pos
    
    def _build_pageviews_by_pos(self, wid2pv) -> np.ndarray:
        """
        Convert doc_id->views dict into a compact array indexed by pos.
        Missing docs get 0.
        """
        pv = np.zeros(self.doc_norm_body.shape[0], dtype=np.uint32)

        for doc_id, views in wid2pv.items():
            # bounds check for doc_id_to_pos
            if doc_id < 0 or doc_id >= self.doc_id_to_pos.shape[0]:
                continue

            pos = self.doc_id_to_pos[doc_id]
            if pos == self.INVALID_POS:
                continue

            # clamp to uint32 range just in case
            v = int(views)
            if v < 0:
                v = 0
            elif v > 2**32 - 1:
                v = 2**32 - 1

            pv[int(pos)] = v

        return pv




    # -------------------------
    # Public API
    # -------------------------
    def get_doc_norm_body(self, doc_id: int) -> float:
        """
        Assumes doc_norm_body is indexed by pos (recommended).
        If yours is indexed by doc_id, change to: return float(self.doc_norm_body[doc_id])
        """
        pos = self._doc_id_to_pos(doc_id)
        if pos is None:
            return 0.0
        return float(self.doc_norm_body[pos])

    def get_inv_doc_len_body(self, doc_id: int) -> float:
        """
        Assumes inv_doc_len_body is indexed by pos.
        """
        pos = self._doc_id_to_pos(doc_id)
        if pos is None:
            return 0.0
        return float(self.inv_doc_len_body[pos])

    def get_title(self, doc_id: int) -> str:
        pos = self._doc_id_to_pos(doc_id)
        if pos is None:
            return ""

        start = int(self.titles_offsets[pos])
        end = int(self.titles_offsets[pos + 1])
        if end <= start:
            return ""

        return self.titles_data[start:end].tobytes().decode("utf-8", errors="replace")
    
    def get_page_rank(self, doc_id: int) -> float:
        """
        Returns PageRank for a doc_id (0.0 if missing / unknown).
        PageRank is stored as an array aligned by pos.
        """
        if self.pagerank_by_pos is None:
            return 0.0
        pos = self._doc_id_to_pos(doc_id)
        if pos is None:
            return 0.0
        return float(self.pagerank_by_pos[pos])
    
    def get_pageviews(self, doc_id: int) -> int:
        """
        Returns pageviews for doc_id (0 if missing / not loaded).
        Uses doc_id->pos mapping + pos-aligned array.
        """
        if self.pageviews_by_pos is None:
            return 0
        pos = self._doc_id_to_pos(doc_id)
        if pos is None:
            return 0
        return int(self.pageviews_by_pos[pos])

    def get_doc_len_body(self, doc_id: int) -> float:
        """
        Returns body doc length for doc_id (0 if missing).
        Uses inv_doc_len_body: doc_len = 1 / inv_len
        """
        inv_len = self.get_inv_doc_len_body(doc_id)
        if inv_len <= 0.0:
            return 0.0
        return 1.0 / inv_len

    def get_avg_doc_len_body(self) -> float:
        """Average body document length (avgdl) computed once at init."""
        return float(self.avg_doc_len_body)


    # -------------------------
    # Private utilities
    # -------------------------
    def _doc_id_to_pos(self, doc_id: int) -> Optional[int]:
        if doc_id < 0 or doc_id >= self.doc_id_to_pos.shape[0]:
            return None

        pos = self.doc_id_to_pos[doc_id]
        if pos == self.INVALID_POS:
            return None

        return int(pos)


    def _compute_avg_doc_len_body(self) -> float:
        """
        Computes average body document length (avgdl) using inv_doc_len_body:
            inv_doc_len_body[pos] = 1 / doc_len
        So doc_len = 1 / inv_len for inv_len > 0.
        """
        inv = self.inv_doc_len_body.astype(np.float64, copy=False)
        mask = inv > 0.0
        if not np.any(mask):
            return 0.0
        lengths = 1.0 / inv[mask]
        return float(lengths.mean())