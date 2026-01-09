# test_body_module.py
# Simple smoke test: run a query and print top 100 (doc_id, score)

import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]  # modules_checks -> src
sys.path.insert(0, str(SRC_DIR))
import re
from body_module import BodyModule, BodyIndexConfig
from meta_data_module import *
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
import json
# nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text: str):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token not in all_stopwords]
    return tokens


def main(query: str):
    # ---- config (EDIT if your paths differ) ----
    BUCKET_NAME = "ir_3_207472234"

    BODY_BASE_DIR = "indexes/postings_gcp"
    BODY_INDEX_NAME = "index"   # loads gs://BUCKET/indexes/body/body_index.pkl
    mode = "gcs"

    # =========================
    # Fill paths
    # For GCS: these are blob paths inside the bucket (NOT full URLs)
    # For local: filesystem paths
    # =========================
    paths = MetaDataPaths(
        doc_id_to_pos="meta_data/doc_id_to_pos.npy",          
        doc_norm_body="meta_data/doc_norm_body.npy",          
        inv_doc_len_body="meta_data/inv_doc_len_body.npy",    
        titles_data="metadata/titles_data.bin",              
        titles_offsets="metadata/titles_offsets.bin",        
        pagerank_csv_gz="pr/part-00000-01ae429d-6dc4-4410-9263-84d031c009d4-c000.csv.gz",
        pageviews_pkl="meta_data/pageviews-202108-user.pkl"
    )

    # =========================
    # Init module
    # =========================
    if mode == "gcs":
        bucket_name = "ir_3_207472234"  # <-- your bucket
        md = MetaDataModule(paths=paths, mode="gcs", bucket_name=bucket_name)
    else:
        md = MetaDataModule(paths=paths, mode="local")

    # ---- load body module ----
    body = BodyModule(
        config=BodyIndexConfig(
            base_dir=BODY_BASE_DIR,
            index_name=BODY_INDEX_NAME,
            mode="gcs",
            bucket_name=BUCKET_NAME,
            is_text_posting=True,
        ),
        meta_data_module=md,
    )

    print("start the real timing")
    t = time()

    # ---- tokenize ----
    terms = tokenize(query)
    if not terms:
        return []

    # ---- search ----
    results = body.search_bm25(terms, max_workers=16, top_k=100)
    print("the retrival took: ", time() - t)
    return results


QUERIES_PATH = "C:\\Users\\moran\\university\\semester 5\\information retrival\\final_project\\queries_train.json"

if __name__ == "__main__":
        # ---- load queries ----
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # ---- take first query ----
    # query = next(iter(queries.keys()))
    query_keys = list(queries.keys())
    for i in range(5):  # first 5 queries
        query = query_keys[i]
        gold_docs = [int(d) for d in queries[query]]
        
        print("Query:", query)
        print("Gold docs count:", len(gold_docs))  # should be 100

        results = main(query)
        pred_docs = [doc_id for doc_id, _ in results]

        # ---- intersection ----
        gold_set = set(gold_docs)
        pred_set = set(pred_docs)

        intersection = gold_set & pred_set


        print("Retrieved docs:", len(pred_docs))  # should be 100
        print("Overlap (intersection) count:", len(intersection))
        print("Recall@100:", len(intersection) / len(gold_set))

        print("\nOverlapping doc_ids (up to 20):")
        print(list(intersection)[:20])

