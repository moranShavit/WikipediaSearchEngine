# import pickle
# import inverted_index_gcp as idx  # this is the file you pasted


# # ====== SET THESE ======
# BUCKET_NAME = "ir_3_207472234"
# # Option A: load using your InvertedIndex.read_index (preferred if it's an index pickle)
# BASE_DIR = "postings_gcp"        # folder/prefix inside the bucket ("" if none)
# INDEX_NAME = "index"   # loads gs://BUCKET/BASE_DIR/my_index.pkl

# # Option B: load any pickle directly by path (use this if you don't know it's an index)
# PICKLE_PATH = "postings_gcp/index.pkl"
# # =======================


# def preview(obj):
#     print("Type:", type(obj))

#     # If it's your InvertedIndex object
#     if isinstance(obj, idx.InvertedIndex):
#         print("\n✅ InvertedIndex loaded")
#         print("Number of terms:", len(obj.df))
#         print("Top df terms:", obj.df.most_common(10))
#         print("Top total terms:", obj.term_total.most_common(10))

#         if len(obj.posting_locs) > 0:
#             t = next(iter(obj.posting_locs.keys()))
#             print("\nSample posting_locs term:", t)
#             print("Locs (first 3):", obj.posting_locs[t][:3])
#         return

#     # dict / list fallback
#     if isinstance(obj, dict):
#         print("\nDict keys (first 20):")
#         for i, k in enumerate(obj.keys()):
#             print(" -", k)
#             if i == 19:
#                 break
#         return

#     if isinstance(obj, (list, tuple)):
#         print("\nLength:", len(obj))
#         print("First 10 items:", obj[:10])
#         return

#     print("\nPreview repr:")
#     s = repr(obj)
#     print(s[:2000] + ("..." if len(s) > 2000 else ""))


# def main():
#     print("=== Option A: Using InvertedIndex.read_index ===")
#     try:
#         index_obj = idx.InvertedIndex.read_index(
#             base_dir=BASE_DIR,
#             name=INDEX_NAME,
#             bucket_name=BUCKET_NAME
#         )
#         print("✅ Loaded with read_index")
#         preview(index_obj)
#         return
#     except Exception as e:
#         print("read_index failed:", repr(e))

#     print("\n=== Option B: Direct pickle load via your _open/get_bucket ===")
#     bucket = idx.get_bucket(BUCKET_NAME)
#     with idx._open(PICKLE_PATH, "rb", bucket) as f:
#         obj = pickle.load(f)

#     print("✅ Loaded with direct pickle load")
#     preview(obj)


# if __name__ == "__main__":
#     main()

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]  # modules_checks -> src
sys.path.insert(0, str(SRC_DIR))

import inverted_index_gcp as idx

BUCKET_NAME = "ir_3_207472234"
BASE_DIR = "indexes/anchor_text_to_linked_page_aggregate"
INDEX_NAME = "anchor_to_linked_page_index"  # loads gs://BUCKET_NAME/postings_gcp/index.pkl


def main():
    print(f"Loading gs://{BUCKET_NAME}/{BASE_DIR}/{INDEX_NAME}.pkl ...")

    index = idx.InvertedIndex.read_index(
        base_dir=BASE_DIR,
        name=INDEX_NAME,
        bucket_name=BUCKET_NAME
    )

    print("\n✅ Loaded index")
    print("Type:", type(index))

    # Basic stats
    print("\n--- Basic stats ---")
    print("Num terms (len(df)):", len(index.df))
    print("Num terms with posting_locs:", len(index.posting_locs))
    print("Num terms in term_total:", len(index.term_total))

    # Top df terms
    print("\n--- Top DF terms ---")
    try:
        print(index.df.most_common(20))
    except Exception as e:
        print("Could not compute most_common on df:", repr(e))

    # Show one sample term and try reading its posting list
    if len(index.posting_locs) > 0:
        # term = next(iter(index.posting_locs.keys()))
        term = "collectivism"
        print("\n--- Sample term ---")
        print("Term:", term)
        print("df[term]:", index.df.get(term))
        print("posting_locs[term] (first 3):", index.posting_locs[term][:3])

        try:
            pl = index.read_a_posting_list(BASE_DIR, term, BUCKET_NAME, is_text_posting=False)
            print("Posting list sample (first 10):", pl[:10])
            for doc, _ in pl:
                if doc == 23040 or doc == "230404":
                    print("you posting is probably ok")
                    break
        except Exception as e:
            print("Could not read posting list for sample term:", repr(e))

    print("\nDone.")


if __name__ == "__main__":
    main()