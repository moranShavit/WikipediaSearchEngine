import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # modules_checks -> src -> project root
sys.path.insert(0, str(PROJECT_ROOT))

import gzip
import io
import csv
from src.meta_data_module import _download_blob_bytes  


def debug_print_pagerank_file(csv_gz_bytes: bytes, n_lines: int = 10):
    print("=== PageRank file preview ===")

    with gzip.GzipFile(fileobj=io.BytesIO(csv_gz_bytes), mode="rb") as gz:
        text = gz.read().decode("utf-8", errors="replace")

    lines = text.splitlines()

    print(f"Total lines: {len(lines)}")
    print(f"Showing first {min(n_lines, len(lines))} lines:\n")

    for i, line in enumerate(lines[:n_lines]):
        print(f"{i}: {line}")

    print("\n=== CSV parsing preview ===")
    reader = csv.reader(lines[:n_lines])
    for i, row in enumerate(reader):
        print(f"row {i}: {row} (cols={len(row)})")




csv_bytes = _download_blob_bytes(
    "ir_3_207472234",
    "pr/part-00000-01ae429d-6dc4-4410-9263-84d031c009d4-c000.csv.gz",
)

debug_print_pagerank_file(csv_bytes, n_lines=10)
