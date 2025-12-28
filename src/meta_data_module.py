# meta_data_module.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
from io import BytesIO
import os

import numpy as np
from google.cloud import storage


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

        elif self.mode == "local":
            # ---- Load numpy arrays into RAM (from local) ----
            self.doc_id_to_pos = self._load_npy_local(paths.doc_id_to_pos)
            self.doc_norm_body = self._load_npy_local(paths.doc_norm_body)
            self.inv_doc_len_body = self._load_npy_local(paths.inv_doc_len_body)

            # ---- Load titles bins into RAM (from local) ----
            offsets_bytes = self._read_file_bytes(paths.titles_offsets)
            data_bytes = self._read_file_bytes(paths.titles_data)

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
