
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]  # modules_checks -> src
sys.path.insert(0, str(SRC_DIR))

from search_engine import *

# eng = SearchEngine("gcs", "ir_3_207472234")


# print(eng.search("Mount Everest climbing expeditions"))

import json
import time
from typing import Dict, List, Tuple, Any, Optional


def _precision_recall_at_k(top_docids: List[int], relevant: set, k: int) -> tuple[float, float, int, int]:
    """
    Returns (precision@k, recall@k, num_found, denom_k_used)
    denom_k_used is the number of retrieved docs considered (<=k if fewer results returned)
    """
    topk = top_docids[:k]
    denom = max(len(topk), 1)  # avoid /0 if no results
    num_found = sum(1 for d in topk if d in relevant)
    precision = num_found / denom
    recall = num_found / max(len(relevant), 1)  # avoid /0 if qrels empty
    return precision, recall, num_found, denom


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def _extract_docid(result_item: Any) -> Optional[int]:
    """
    Your engine returns items like: (50651770, 'Title').
    This safely extracts the docid as int.
    """
    if isinstance(result_item, (list, tuple)) and len(result_item) >= 1:
        try:
            return int(result_item[0])
        except (TypeError, ValueError):
            return None
    # If someday it returns a dict like {"docid": ..., "title": ...}
    if isinstance(result_item, dict) and "docid" in result_item:
        try:
            return int(result_item["docid"])
        except (TypeError, ValueError):
            return None
    return None


# def evaluate_search_engine(
#     eng,
#     qrels_path: str,
#     k: int = 10,
# ) -> Dict[str, Any]:
#     """
#     Returns:
#       {
#         "per_query": {
#            query: {
#              "time_ms": float,
#              "precision@k": float,
#              "recall@k": float,
#              "accuracy@k": int,  # hit@k
#              "num_relevant_total": int,
#              "num_relevant_found_in_topk": int,
#              "topk_docids": [int, ...],
#            }, ...
#         },
#         "macro_avg": {
#            "avg_time_ms": float,
#            "precision@k": float,
#            "recall@k": float,
#            "accuracy@k": float
#         }
#       }
#     """
#     with open(qrels_path, "r", encoding="utf-8") as f:
#         qrels_raw: Dict[str, List[str]] = json.load(f)

#     per_query: Dict[str, Any] = {}

#     sum_time = 0.0
#     sum_prec = 0.0
#     sum_rec = 0.0
#     sum_acc = 0.0
#     n = 0

#     for query, rel_list in qrels_raw.items():
#         # qrels in your JSON are strings → convert to ints
#         relevant = {int(x) for x in rel_list}

#         t0 = time.perf_counter()
#         results = eng.search(query)
#         t1 = time.perf_counter()
#         time_ms = (t1 - t0) * 1000.0

#         # take top-k docids
#         topk_docids: List[int] = []
#         for item in results[:k]:
#             docid = _extract_docid(item)
#             if docid is not None:
#                 topk_docids.append(docid)

#         found_relevant = [d for d in topk_docids if d in relevant]
#         num_found = len(found_relevant)
#         num_rel_total = len(relevant)

#         precision_k = num_found / max(len(topk_docids), 1)          # avoid /0 if no results
#         recall_k = num_found / max(num_rel_total, 1)                # avoid /0 if qrels empty
#         accuracy_k = 1 if num_found > 0 else 0                      # hit@k

#         per_query[query] = {
#             "time_ms": time_ms,
#             "precision@k": precision_k,
#             "recall@k": recall_k,
#             "accuracy@k": accuracy_k,
#             "num_relevant_total": num_rel_total,
#             "num_relevant_found_in_topk": num_found,
#             "topk_docids": topk_docids,
#         }

#         sum_time += time_ms
#         sum_prec += precision_k
#         sum_rec += recall_k
#         sum_acc += accuracy_k
#         n += 1

#     macro_avg = {
#         "avg_time_ms": (sum_time / n) if n else 0.0,
#         "precision@k": (sum_prec / n) if n else 0.0,
#         "recall@k": (sum_rec / n) if n else 0.0,
#         "accuracy@k": (sum_acc / n) if n else 0.0,
#     }

#     return {"per_query": per_query, "macro_avg": macro_avg}


# # ---- example usage ----
# eng = SearchEngine("gcs", "ir_3_207472234")
# report = evaluate_search_engine(eng, "C:\\Users\\moran\\university\\semester 5\\information retrival\\final_project\\queries_train.json", k=100)
# print(report["macro_avg"])

# # ---- per-query performance output ----
# for query, stats in report["per_query"].items():
#     print("=" * 80)
#     print(f"Query: {query}")
#     print(f"Time: {stats['time_ms']:.2f} ms")
#     print(f"Precision@10: {stats['precision@k']:.4f}")
#     print(f"Recall@10:    {stats['recall@k']:.4f}")
#     print(f"Accuracy@10:  {stats['accuracy@k']}")
#     print(
#         f"Relevant found: {stats['num_relevant_found_in_topk']} / "
#         f"{stats['num_relevant_total']}"
#     )
#     print(f"Top-10 docids: {stats['topk_docids']}")

def evaluate_search_engine(
    eng,
    qrels_path: str,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Adds:
      - precision@5
      - precision@30, recall@30, f1@30

    Still returns your original macro_avg for @k plus new macro averages.
    """
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels_raw: Dict[str, List[str]] = json.load(f)

    per_query: Dict[str, Any] = {}

    sum_time = 0.0

    # sums for @k (existing)
    sum_prec_k = 0.0
    sum_rec_k = 0.0
    sum_acc_k = 0.0

    # sums for new metrics
    sum_prec_5 = 0.0
    sum_prec_30 = 0.0
    sum_rec_30 = 0.0
    sum_f1_30 = 0.0

    n = 0

    for query, rel_list in qrels_raw.items():
        relevant = {int(x) for x in rel_list}

        t0 = time.perf_counter()
        results = eng.search(query)
        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0

        # extract docids from ALL results (we'll slice as needed)
        all_docids: List[int] = []
        for item in results:
            docid = _extract_docid(item)
            if docid is not None:
                all_docids.append(docid)

        # existing @k
        precision_k, recall_k, num_found_k, _ = _precision_recall_at_k(all_docids, relevant, k)
        accuracy_k = 1 if num_found_k > 0 else 0  # hit@k

        # Precision@5
        precision_5, _, num_found_5, denom_5 = _precision_recall_at_k(all_docids, relevant, 5)

        # Precision/Recall/F1@30
        precision_30, recall_30, num_found_30, denom_30 = _precision_recall_at_k(all_docids, relevant, 30)
        f1_30 = _f1(precision_30, recall_30)

        per_query[query] = {
            "time_ms": time_ms,

            # original outputs
            f"precision@{k}": precision_k,
            f"recall@{k}": recall_k,
            f"accuracy@{k}": accuracy_k,

            # new outputs
            "precision@5": precision_5,
            "precision@30": precision_30,
            "recall@30": recall_30,
            "f1@30": f1_30,

            "num_relevant_total": len(relevant),

            # counts are sometimes useful for debugging
            f"num_relevant_found_in_top{k}": num_found_k,
            "num_relevant_found_in_top5": num_found_5,
            "num_relevant_found_in_top30": num_found_30,

            # keep docids for the original view
            f"top{k}_docids": all_docids[:k],
            "top5_docids": all_docids[:5],
            "top30_docids": all_docids[:30],

            # denominators actually used (if fewer docs returned than k)
            f"denom_used@{k}": min(len(all_docids), k) if len(all_docids) > 0 else 0,
            "denom_used@5": denom_5 if len(all_docids) > 0 else 0,
            "denom_used@30": denom_30 if len(all_docids) > 0 else 0,
        }

        sum_time += time_ms

        sum_prec_k += precision_k
        sum_rec_k += recall_k
        sum_acc_k += accuracy_k

        sum_prec_5 += precision_5
        sum_prec_30 += precision_30
        sum_rec_30 += recall_30
        sum_f1_30 += f1_30

        n += 1

    macro_avg = {
        "avg_time_ms": (sum_time / n) if n else 0.0,

        # macro averages for original @k
        f"precision@{k}": (sum_prec_k / n) if n else 0.0,
        f"recall@{k}": (sum_rec_k / n) if n else 0.0,
        f"accuracy@{k}": (sum_acc_k / n) if n else 0.0,

        # macro averages for new metrics
        "precision@5": (sum_prec_5 / n) if n else 0.0,
        "precision@30": (sum_prec_30 / n) if n else 0.0,
        "recall@30": (sum_rec_30 / n) if n else 0.0,
        "f1@30": (sum_f1_30 / n) if n else 0.0,
    }

    return {"per_query": per_query, "macro_avg": macro_avg}


# ---- example usage ----
eng = SearchEngine("gcs", "ir_3_207472234")
report = evaluate_search_engine(eng, "C:\\Users\\moran\\university\\semester 5\\information retrival\\final_project\\queries_train.json", k=100)
print(report["macro_avg"])



# ---- per-query performance output ----
for query, stats in report["per_query"].items():
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Time: {stats['time_ms']:.2f} ms")

    print(f"Precision@5:  {stats['precision@5']:.4f}")
    print(f"F1@30:        {stats['f1@30']:.4f}  (P@30={stats['precision@30']:.4f}, R@30={stats['recall@30']:.4f})")

    print(f"Precision@100: {stats['precision@100']:.4f}")
    print(f"Recall@100:    {stats['recall@100']:.4f}")
    print(f"Accuracy@100:  {stats['accuracy@100']}")

    print(
        f"Relevant found in top100: {stats['num_relevant_found_in_top100']} / "
        f"{stats['num_relevant_total']}"
    )
    print(f"Top-5 docids:  {stats['top5_docids']}")
    print(f"Top-30 docids: {stats['top30_docids']}")
    print(f"Top-100 docids:{stats['top100_docids']}")