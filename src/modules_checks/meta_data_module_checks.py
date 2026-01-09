# test_meta_data_module.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # modules_checks -> src -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from src.meta_data_module import MetaDataModule, MetaDataPaths

def main():
    # =========================
    # Choose mode: "gcs" or "local"
    # =========================
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

    # =========================
    # Quick sanity prints
    # =========================
    print("=== Loaded MetaDataModule ===")
    print("doc_id_to_pos:", md.doc_id_to_pos.shape, md.doc_id_to_pos.dtype)
    print("doc_norm_body:", md.doc_norm_body.shape, md.doc_norm_body.dtype)
    print("inv_doc_len_body:", md.inv_doc_len_body.shape, md.inv_doc_len_body.dtype)
    print("titles_offsets:", md.titles_offsets.shape, md.titles_offsets.dtype)
    print("titles_data:", md.titles_data.shape, md.titles_data.dtype)
    print("INVALID_POS sentinel:", md.INVALID_POS)

    # ✅ PageRank sanity
    has_pr = hasattr(md, "pagerank_by_pos") and (md.pagerank_by_pos is not None)
    print("pagerank loaded:", has_pr)
    if has_pr:
        print("pagerank_by_pos:", md.pagerank_by_pos.shape, md.pagerank_by_pos.dtype)

    # =========================
    # Try a few doc_ids
    # You can change these to docs you know exist
    # =========================
    test_doc_ids = [12, 25, 39, 290, 303,305]

    print("\n=== Sample lookups ===")
    for doc_id in test_doc_ids:
        # show pos (private helper exists, but we'll access safely using the public getters)
        title = md.get_title(doc_id)
        norm = md.get_doc_norm_body(doc_id)
        inv_len = md.get_inv_doc_len_body(doc_id)

        print(f"\nDocID: {doc_id}")
        print("  title:", repr(title[:120] + ("..." if len(title) > 120 else "")))
        print("  doc_norm_body:", norm)
        print("  inv_doc_len_body:", inv_len)

    # =========================
    # Optional: a small consistency check for titles offsets
    # (Only runs if doc_id_to_pos has enough range)
    # =========================
    print("\n=== Offset consistency check (first few valid positions) ===")
    checked = 0
    for doc_id in range(min(1000, md.doc_id_to_pos.shape[0])):
        # replicate internal mapping safely:
        pos = md.doc_id_to_pos[doc_id]
        if pos == md.INVALID_POS:
            continue
        pos = int(pos)
        if pos + 1 >= md.titles_offsets.shape[0]:
            continue
        start = int(md.titles_offsets[pos])
        end = int(md.titles_offsets[pos + 1])
        if end < start:
            print(f"  BAD OFFSETS at pos={pos} (start={start}, end={end})")
            break
        checked += 1
        if checked >= 5:
            break
    print(f"  checked {checked} positions (no obvious offset issues).")

    # doc_id -> expected pagerank (from the photo)
    expected = {
        3434750: 9913.728782160782,
        10568:   5385.349263642039,
        32927:   5282.081575765278,
        30680:   5128.23370960412,
        5843419: 4957.567686263868,
        68253:   4769.278265355159,
        31717:   4486.35018054831,
        11867:   4146.414650912771,
        14533:   3996.4664408855033,
        17867:   3246.0983906041424,
        5042916: 2991.945739166179,
        4689264: 2982.324883041748,
        14532:   2934.7468292031717,
        25391:   2903.5462235133987,
        5405:    2891.4163291546374,
        4764461: 2834.366987332662,
        15573:   2783.865118588384,
        9316:    2782.0396464137707,
        8569916: 2775.2861918400163,
    }

    print("=== PageRank comparison ===")
    all_ok = True

    for doc_id, exp_pr in expected.items():
        got_pr = md.get_page_rank(doc_id)
        ok = approx_equal(got_pr, exp_pr)

        status = "EQUAL ✅" if ok else "NOT EQUAL ❌"
        print(f"doc_id={doc_id:8d}  expected={exp_pr:.6f}  got={got_pr:.6f}  -> {status}")

        if not ok:
            all_ok = False

    print("\nFINAL RESULT:", "ALL MATCH ✅" if all_ok else "MISMATCHES FOUND ❌")

    temp = md.get_pageviews(17324616)
    if temp == 10:
        print("page views getter works!!!")
    else:
        print(f"there is a problem in page vies getter it return {temp} insted of 10")

def approx_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps * max(1.0, abs(a), abs(b))   

if __name__ == "__main__":
    main()
