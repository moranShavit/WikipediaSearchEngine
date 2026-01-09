# optimize_bm25.py
import json
import csv
import math
import re
import argparse
from time import time
from typing import List, Dict, Tuple, Iterable, Set
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]  # modules_checks -> src
sys.path.insert(0, str(SRC_DIR))

# ----------------------------
# Tokenization (same idea as yours)
# ----------------------------
def build_tokenizer():
    try:
        from nltk.corpus import stopwords
        english_stopwords = frozenset(stopwords.words("english"))
    except Exception:
        # # If NLTK stopwords aren't available, fallback to a small list
        # english_stopwords = frozenset({
        #     "a","an","the","and","or","to","of","in","on","for","with","by","is","are"
        # })
        print("stop words load has failed!!!")

    corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]
    all_stopwords = english_stopwords.union(corpus_stopwords)

    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    def tokenize(text: str) -> List[str]:
        tokens = []
        for m in RE_WORD.finditer(text.lower()):
            tok = m.group()
            if tok not in all_stopwords:
                tokens.append(tok)
        return tokens

    return tokenize

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

def f1_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    topk = ranked[:k]
    if not topk or not relevant:
        return 0.0
    hits = sum(1 for d in topk if d in relevant)
    prec = hits / float(k)
    rec = hits / float(k)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)

def harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)

# ----------------------------
# Run one configuration
# ----------------------------
def eval_config(
    body,
    queries: List[Tuple[str, List[int]]],
    tokenize,
    *,
    k1: float,
    b: float,
    use_bm25plus: bool,
    delta: float,
    top_k_retrieve: int = 100,
    max_workers: int = 16,
) -> Dict[str, float]:
    p5_list = []
    f1_30_list = []
    hm_list = []

    for q, rel_list in queries:
        relevant = set(rel_list)
        terms = tokenize(q)
        if not terms:
            p5_list.append(0.0)
            f1_30_list.append(0.0)
            hm_list.append(0.0)
            continue

        # BodyModule.search_bm25 supports k1, b, use_bm25plus, delta :contentReference[oaicite:3]{index=3}
        results = body.search_bm25(
            terms,
            max_workers=max_workers,
            top_k=top_k_retrieve,
            k1=k1,
            b=b,
            use_bm25plus=use_bm25plus,
            delta=delta,
        )

        ranked_docids = [int(doc_id) for doc_id, _ in results]

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
    }

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries_json", default="/home/moran/ir_project/final_project/queries_train.json")
    ap.add_argument("--out_csv", default="bm25_grid_results_v0.csv")
    ap.add_argument("--train_n", type=int, default=24)

    # Grids (edit as you like)
    ap.add_argument("--k1_values", default="0.6,0.9,1.2,1.5,1.8,2.1,2.4")
    ap.add_argument("--b_values", default="0.0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--delta_values", default="0.0,0.5,1.0,1.5,2.0,3.0")

    # retrieval params
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--max_workers", type=int, default=16)

    args = ap.parse_args()

    # ---------- load training queries ----------
    with open(args.queries_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data is dict: query -> [doc_id strings] :contentReference[oaicite:4]{index=4}
    items = list(data.items())[: args.train_n]
    train_queries: List[Tuple[str, List[int]]] = []
    for q, rels in items:
        train_queries.append((q, [int(x) for x in rels]))

    tokenize = build_tokenizer()

    # ---------- INIT YOUR BODY MODULE HERE ----------
    # You said you already do something like this:
    #
    #   md = MetaDataModule(paths=paths, mode="gcs", bucket_name=BUCKET_NAME)
    #   body = BodyModule(config=BodyIndexConfig(...), meta_data_module=md)
    #
    # Fill this section with your real init code.
    from body_module import BodyModule, BodyIndexConfig  # :contentReference[oaicite:5]{index=5}

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # FILL HERE (same as your snippet)
    BUCKET_NAME = "ir_3_207472234"
    BODY_BASE_DIR = "indexes/postings_gcp"
    BODY_INDEX_NAME = "index"
    mode = "gcs"

    # If you have these in your project, import them:
    from meta_data_module import MetaDataModule, MetaDataPaths
    
    paths = MetaDataPaths(
        doc_id_to_pos="meta_data/doc_id_to_pos.npy",          
        doc_norm_body="meta_data/doc_norm_body.npy",          
        inv_doc_len_body="meta_data/inv_doc_len_body.npy",    
        titles_data="metadata/titles_data.bin",              
        titles_offsets="metadata/titles_offsets.bin",        
        pagerank_csv_gz="pr/part-00000-01ae429d-6dc4-4410-9263-84d031c009d4-c000.csv.gz",
        pageviews_pkl="meta_data/pageviews-202108-user.pkl"
    )
    md = MetaDataModule(paths=paths, mode="gcs", bucket_name=BUCKET_NAME)
    
    body = BodyModule(
        config=BodyIndexConfig(
            base_dir=BODY_BASE_DIR,
            index_name=BODY_INDEX_NAME,
            mode=mode,
            bucket_name=BUCKET_NAME,
            is_text_posting=True,
        ),
        meta_data_module=md,
    )

    # ---------- parse grids ----------
    k1_values = [float(x) for x in args.k1_values.split(",")]
    b_values = [float(x) for x in args.b_values.split(",")]
    delta_values = [float(x) for x in args.delta_values.split(",")]

    # ---------- grid search + CSV ----------
    fieldnames = [
        "model", "k1", "b", "delta",
        "mean_p5", "mean_f1_30", "mean_hmean",
        "train_n", "top_k", "max_workers",
        "seconds"
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        # BM25 (delta ignored)
        for k1 in k1_values:
            for b in b_values:
                t0 = time()
                metrics = eval_config(
                    body,
                    train_queries,
                    tokenize,
                    k1=k1,
                    b=b,
                    use_bm25plus=False,
                    delta=0.0,
                    top_k_retrieve=args.top_k,
                    max_workers=args.max_workers,
                )
                dt = time() - t0
                writer.writerow({
                    "model": "bm25",
                    "k1": k1,
                    "b": b,
                    "delta": "",
                    **metrics,
                    "train_n": args.train_n,
                    "top_k": args.top_k,
                    "max_workers": args.max_workers,
                    "seconds": round(dt, 4),
                })
                fcsv.flush()

        # BM25+
        for k1 in k1_values:
            for b in b_values:
                for delta in delta_values:
                    t0 = time()
                    metrics = eval_config(
                        body,
                        train_queries,
                        tokenize,
                        k1=k1,
                        b=b,
                        use_bm25plus=True,
                        delta=delta,
                        top_k_retrieve=args.top_k,
                        max_workers=args.max_workers,
                    )
                    dt = time() - t0
                    writer.writerow({
                        "model": "bm25plus",
                        "k1": k1,
                        "b": b,
                        "delta": delta,
                        **metrics,
                        "train_n": args.train_n,
                        "top_k": args.top_k,
                        "max_workers": args.max_workers,
                        "seconds": round(dt, 4),
                    })
                    fcsv.flush()

    print(f"Done. Wrote results to: {args.out_csv}")

if __name__ == "__main__":
    main()
