"""
Grid search over hybrid weight dictionary for SearchEngine.search().

Optimizes:
  harmonic_mean( macro_precision@5, macro_f1@30 )

Writes one row per configuration to CSV, plus a "best" row at the end.

Usage example:
  python grid_search_weights.py \
    --mode gcs --bucket ir_3_207472234 \
    --queries_json /home/moran/ir_project/final_project/queries_train.json \
    --train_n 24 \
    --out_csv weights_grid_results.csv \
    --w_body "0.5,1.0,1.5" \
    --w_title "0.0,0.5,1.0" \
    --w_anchor "0.0,0.5,1.0" \
    --w_pr "0.0,0.2,0.4" \
    --w_pv "0.0,0.2,0.4"
"""

import argparse
import csv
import itertools
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]  # modules_checks -> src
sys.path.insert(0, str(SRC_DIR))


# ----------------------------
# Metrics
# ----------------------------
def precision_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in relevant)
    return hits / float(k)

def recall_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / float(len(relevant))

def f1_from_pr(p: float, r: float) -> float:
    return (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0

def f1_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    return f1_from_pr(p, r)

def harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)

# ----------------------------
# Helpers
# ----------------------------
def parse_float_list(s: str) -> List[float]:
    # allow "0.1,0.2,0.3" or "0.1 0.2 0.3"
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip() != ""]
    return [float(x) for x in parts]

def extract_docid(result_item: Any) -> Optional[int]:
    # Your engine returns (docid, title) tuples. Keep this robust anyway.
    if isinstance(result_item, (list, tuple)) and len(result_item) >= 1:
        try:
            return int(result_item[0])
        except (TypeError, ValueError):
            return None
    if isinstance(result_item, dict) and "docid" in result_item:
        try:
            return int(result_item["docid"])
        except (TypeError, ValueError):
            return None
    return None

@dataclass(frozen=True)
class WeightConfig:
    body: float
    title: float
    anchor: float
    pagerank: float
    pageviews: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "body": self.body,
            "title": self.title,
            "anchor": self.anchor,
            "pagerank": self.pagerank,
            "pageviews": self.pageviews,
        }

# ----------------------------
# Evaluate one configuration
# ----------------------------
def eval_config(
    eng,
    queries: List[Tuple[str, List[int]]],
    weights: Dict[str, float],
    *,
    top_k: int,
    body_k: int,
    title_k: int,
    anchor_k: int,
    max_workers: int,
    use_log_for_views: bool,
) -> Dict[str, float]:
    p5_list: List[float] = []
    f1_30_list: List[float] = []
    hm_list: List[float] = []
    time_ms_list: List[float] = []

    for q, rel_list in queries:
        relevant = set(int(x) for x in rel_list)

        t0 = time.perf_counter()
        results = eng.search(
            q,
            top_k=top_k,
            body_k=body_k,
            title_k=title_k,
            anchor_k=anchor_k,
            max_workers=max_workers,
            weights=weights,
            use_log_for_views=use_log_for_views,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        time_ms_list.append(dt_ms)

        ranked_docids: List[int] = []
        for item in results:
            did = extract_docid(item)
            if did is not None:
                ranked_docids.append(did)

        p5 = precision_at_k(ranked_docids, relevant, 5)
        f1_30 = f1_at_k(ranked_docids, relevant, 30)
        hm = harmonic_mean(p5, f1_30)

        p5_list.append(p5)
        f1_30_list.append(f1_30)
        hm_list.append(hm)

    def avg(xs: List[float]) -> float:
        return sum(xs) / float(len(xs)) if xs else 0.0

    return {
        "mean_p5": avg(p5_list),
        "mean_f1_30": avg(f1_30_list),
        "mean_hmean": avg(hm_list),
         "avg_time_ms": avg(time_ms_list),
    }

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gcs", choices=["gcs", "local"])
    ap.add_argument("--bucket", default="ir_3_207472234")
    ap.add_argument("--queries_json", required=True, default="/home/moran/ir_project/final_project/queries_train.json")
    ap.add_argument("--out_csv", default="weights_grid_results.csv")
    ap.add_argument("--train_n", type=int, default=24)

    # SearchEngine.search retrieval knobs (keep fixed while tuning weights)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--body_k", type=int, default=400)
    ap.add_argument("--title_k", type=int, default=400)
    ap.add_argument("--anchor_k", type=int, default=400)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--use_log_for_views", action="store_true", default=True)
    ap.add_argument("--no_log_for_views", action="store_true", default=False)

    # Weight grids
    ap.add_argument("--w_body", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--w_title", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--w_anchor", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--w_pr", default="0.0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--w_pv", default="0.0,0.1,0.2,0.3,0.4,0.5")

    args = ap.parse_args()
    use_log = args.use_log_for_views and not args.no_log_for_views

    # ---------- load queries ----------
    with open(args.queries_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())[: args.train_n]
    train_queries: List[Tuple[str, List[int]]] = []
    for q, rels in items:
        train_queries.append((q, [int(x) for x in rels]))

    # ---------- init engine ----------
    # IMPORTANT: this imports your SearchEngine implementation.
    # Ensure your PYTHONPATH includes the folder containing search_engine.py.
    from search_engine import SearchEngine  # uses your file's SearchEngine :contentReference[oaicite:1]{index=1}

    eng = SearchEngine(args.mode, args.bucket)

    # ---------- parse grids ----------
    grid_body = parse_float_list(args.w_body)
    grid_title = parse_float_list(args.w_title)
    grid_anchor = parse_float_list(args.w_anchor)
    grid_pr = parse_float_list(args.w_pr)
    grid_pv = parse_float_list(args.w_pv)

    configs = [
        WeightConfig(b, t, a, pr, pv)
        for (b, t, a, pr, pv) in itertools.product(grid_body, grid_title, grid_anchor, grid_pr, grid_pv)
    ]

    fieldnames = [
        "w_body", "w_title", "w_anchor", "w_pagerank", "w_pageviews",
        "mean_p5", "mean_f1_30", "mean_hmean", "avg_time_ms",
        "train_n", "top_k", "body_k", "title_k", "anchor_k", "max_workers",
        "use_log_for_views",
        "seconds_total",
    ]

    best_row: Optional[Dict[str, Any]] = None
    t_all = time.time()

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for i, cfg in enumerate(configs, start=1):
            t0 = time.time()
            weights = cfg.to_dict()

            metrics = eval_config(
                eng,
                train_queries,
                weights,
                top_k=args.top_k,
                body_k=args.body_k,
                title_k=args.title_k,
                anchor_k=args.anchor_k,
                max_workers=args.max_workers,
                use_log_for_views=use_log,
            )

            dt = time.time() - t0
            row = {
                "w_body": cfg.body,
                "w_title": cfg.title,
                "w_anchor": cfg.anchor,
                "w_pagerank": cfg.pagerank,
                "w_pageviews": cfg.pageviews,
                **metrics,
                "train_n": args.train_n,
                "top_k": args.top_k,
                "body_k": args.body_k,
                "title_k": args.title_k,
                "anchor_k": args.anchor_k,
                "max_workers": args.max_workers,
                "use_log_for_views": int(use_log),
                "seconds_total": round(dt, 4),
            }

            writer.writerow(row)
            fcsv.flush()

            if best_row is None or row["mean_hmean"] > best_row["mean_hmean"]:
                best_row = row

        # Append a final "best" row (easy to spot in CSV)
        if best_row is not None:
            writer.writerow({k: "" for k in fieldnames})
            best_marker = dict(best_row)
            best_marker["w_body"] = f"BEST: {best_marker['w_body']}"
            writer.writerow(best_marker)

    print(f"Done. Wrote results to: {args.out_csv}")
    if best_row:
        print("Best config (by mean_hmean):")
        print(best_row)

if __name__ == "__main__":
    main()