# search_engine.py (or engine_search.py)

from meta_data_module import MetaDataModule, MetaDataPaths
from body_module import BodyModule, BodyIndexConfig
from title_module import TitleModule, TitleIndexConfig
from anchor_module import AnchorIndexConfig, AnchorModule
import re
from nltk.corpus import stopwords
from typing import *
from collections import defaultdict
import heapq
import math
from concurrent.futures import ThreadPoolExecutor

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = {
    "category", "references", "also", "external", "links",
    "may", "first", "see", "history", "people", "one", "two",
    "part", "thumb", "including", "second", "following",
    "many", "however", "would", "became",
}

ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)



class SearchEngine:
    def __init__(self, mode: str, bucket_name: str):
        """
        mode: "gcs" or "local"
        """

        # -------------------------
        # MetaDataModule
        # -------------------------
        paths = MetaDataPaths(
        doc_id_to_pos="meta_data/doc_id_to_pos.npy",          
        doc_norm_body="meta_data/doc_norm_body.npy",          
        inv_doc_len_body="meta_data/inv_doc_len_body.npy",    
        titles_data="metadata/titles_data.bin",              
        titles_offsets="metadata/titles_offsets.bin",        
        pagerank_csv_gz="pr/part-00000-01ae429d-6dc4-4410-9263-84d031c009d4-c000.csv.gz",
        pageviews_pkl="meta_data/pageviews-202108-user.pkl"
    )

        if mode == "gcs":
            self.meta = MetaDataModule(paths=paths, mode="gcs", bucket_name=bucket_name)
        else:
            self.meta = MetaDataModule(paths=paths, mode="local")

        # -------------------------
        # BodyModule
        # -------------------------
        body_config = BodyIndexConfig(
            base_dir="indexes/postings_gcp",
            index_name="index",
            mode=mode,
            bucket_name=bucket_name if mode == "gcs" else None,
            is_text_posting=True,
        )
        self.body_module = BodyModule(config=body_config, meta_data_module=self.meta)

        # -------------------------
        # TitleModule (title index)
        # -------------------------
        title_config = TitleIndexConfig(
            base_dir="indexes/title",      
            index_name="title_index",      
            mode=mode,
            bucket_name=bucket_name if mode == "gcs" else None,
            is_text_posting=False,
        )
        self.title_module = TitleModule(config=title_config)

        # -------------------------
        # AnchorModule = TitleModule configured on anchor index
        # (per your request: use TitleModule as the inner anchor module)
        # -------------------------
        anchor_config = TitleIndexConfig(
            base_dir="indexes/anchor_text",  
            index_name="anchor_index",      
            mode=mode,
            bucket_name=bucket_name if mode == "gcs" else None,
            is_text_posting=False,
        )
        self.inner_anchor_module = TitleModule(config=anchor_config)

       # -------------------------
        # AnchorModule (anchor index)
        # -------------------------
        title_config = AnchorIndexConfig(
            base_dir="indexes/anchor_text_to_linked_page_aggregate",      
            index_name="anchor_to_linked_page_index",      
            mode=mode,
            bucket_name=bucket_name if mode == "gcs" else None,
            is_text_posting=False,
        )
        self.anchor_module = TitleModule(config=title_config) 


    def tokenize(self, text: str):
        """
        Tokenizer used by ALL search methods.
        - lowercase
        - regex based
        - remove stopwords
        - NO stemming
        """
        return [
            token.group()
            for token in RE_WORD.finditer(text.lower())
            if token.group() not in ALL_STOPWORDS
        ]
    
    def search_body(self, query: str, top_k: int = 100) -> List[Tuple[int, str]]:
        """
        /search_body endpoint:
        TF-IDF cosine similarity on body.
        Returns: List[(wiki_id, title)]
        """

        # 1) tokenize query
        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        # 2) score documents using BodyModule
        ranked = self.body_module.search(
            query_terms=query_terms,
            top_k=top_k,
        )  # List[(doc_id, score)]

        if not ranked:
            return []

        # 3) attach titles
        results: List[Tuple[int, str]] = []
        for doc_id, _score in ranked:
            title = self.meta.get_title(doc_id)
            results.append((doc_id, title))

        return results


    def search_title(self, query: str) -> List[Tuple[int, str]]:
        """
        /search_title endpoint:
        - match query words in TITLE
        - sorted by number of DISTINCT query terms matched (desc)
        - return ALL results
        """

        # 1) tokenize query
        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        # 2) title module search
        # returns: List[(doc_id, matched_distinct_terms)]
        ranked = self.title_module.search(query_terms)

        if not ranked:
            return []

        # 3) attach titles
        results: List[Tuple[int, str]] = []
        for doc_id, _match_count in ranked:
            title = self.meta.get_title(doc_id)
            results.append((doc_id, title))

        return results
    

    def search_anchor(self, query: str) -> List[Tuple[int, str]]:
        """
        /search_anchor endpoint:
        - match query words in ANCHOR TEXT
        - sorted by number of DISTINCT query terms matched (desc)
        - return ALL results
        """

        # 1) tokenize query
        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        # 2) anchor module search (anchor_module is TitleModule configured for anchors)
        ranked = self.inner_anchor_module.search(query_terms)  # List[(doc_id, matched_distinct_terms)]

        if not ranked:
            return []

        # 3) attach titles
        results: List[Tuple[int, str]] = []
        for doc_id, _match_count in ranked:
            title = self.meta.get_title(doc_id)
            results.append((doc_id, title))

        return results
    

    def get_pagerank(self, wiki_ids: Iterable[int]) -> List[float]:
        """
        /get_pagerank endpoint: return PageRank values per wiki_id.
        """
        return [float(self.meta.get_page_rank(int(wid))) for wid in wiki_ids]


    def get_pageview(self, wiki_ids: Iterable[int]) -> List[int]:
        """
        /get_pageview endpoint: return August 2021 pageviews per wiki_id.
        MetaDataModule method is named get_pageviews (plural). :contentReference[oaicite:2]{index=2}
        """
        return [int(self.meta.get_pageviews(int(wid))) for wid in wiki_ids]
    

    def search(
        self,
        query: str,
        *,
        top_k: int = 100,
        # candidate cutoffs (speed/recall tradeoff)
        body_k: int = 400,
        title_k: int = 400,
        anchor_k: int = 400,
        max_workers: int = 16,
        # weights to tune on train data
        weights: Optional[Dict[str, float]] = None,
        # normalization choice
        use_log_for_views: bool = True,
    ) -> List[Tuple[int, str]]:
        """
        Hybrid search:
          score(doc) = w_body*norm(body) + w_title*norm(title) + w_anchor*norm(anchor)
                     + w_pr*norm(pagerank) + w_pv*norm(pageviews)

        Returns: top_k list of (wiki_id, title)
        """

        q_terms = self.tokenize(query)
        if not q_terms:
            return []

        # Default weights 
        w = {
            "body": 1.5,
            "title": 0.6,
            "anchor": 0.25,
            "pagerank": 0.1,
            "pageviews": 0.5,
        }
        if weights:
            w.update(weights)

        # -------------------------
        # Stage 1: retrieval signals in parallel
        # -------------------------
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_body = ex.submit(self.body_module.search_bm25, q_terms, max_workers, body_k)
            fut_title = ex.submit(self.title_module.search, q_terms, max_workers)
            fut_anchor = ex.submit(self.anchor_module.search, q_terms, max_workers)

            body_ranked: List[Tuple[int, float]] = fut_body.result() or []
            title_ranked: List[Tuple[int, int]] = fut_title.result() or []
            anchor_ranked: List[Tuple[int, int]] = fut_anchor.result() or []

        # Cut title/anchor to keep candidate set bounded for speed
        if title_k and title_k > 0:
            title_ranked = title_ranked[:title_k]
        if anchor_k and anchor_k > 0:
            anchor_ranked = anchor_ranked[:anchor_k]

        # Turn into dicts: doc_id -> raw signal value
        body_scores = {int(d): float(s) for d, s in body_ranked}
        title_scores = {int(d): float(c) for d, c in title_ranked}
        anchor_scores = {int(d): float(c) for d, c in anchor_ranked}

        # Candidate set = union of all docs seen in any signal
        candidates = set(body_scores) | set(title_scores) | set(anchor_scores)
        if not candidates:
            return []

        # -------------------------
        # Stage 2: metadata signals (pagerank/pageviews)
        # -------------------------
        # Fast O(1) lookups per doc, but we still do it in parallel to overlap Python overhead
        cand_list = list(candidates)

        def _get_pr_map(ids: List[int]) -> Dict[int, float]:
            out = {}
            for d in ids:
                try:
                    out[d] = math.log1p(float(self.meta.get_page_rank(d)))
                except Exception:
                    out[d] = 0.0
            return out

        def _get_pv_map(ids: List[int]) -> Dict[int, float]:
            out = {}
            for d in ids:
                try:
                    pv = float(self.meta.get_pageviews(d))  # meta method name is plural
                except Exception:
                    pv = 0.0
                if use_log_for_views:
                    pv = math.log1p(pv)  # compress heavy tail
                out[d] = pv
            return out

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_pr = ex.submit(_get_pr_map, cand_list)
            fut_pv = ex.submit(_get_pv_map, cand_list)
            pr_scores = fut_pr.result()
            pv_scores = fut_pv.result()

        # -------------------------
        # Normalize each signal to [0,1] over candidates
        # -------------------------
        def _minmax_norm(score_map: Dict[int, float]) -> Dict[int, float]:
            if not score_map:
                return {}
            vals = list(score_map.values())
            mn, mx = min(vals), max(vals)
            if mx == mn:
                # all same -> either all 0 or all 1; choose 0 to avoid adding constant bias
                return {d: 0.0 for d in score_map}
            scale = mx - mn
            return {d: (v - mn) / scale for d, v in score_map.items()}

        n_body = _minmax_norm(body_scores)
        n_title = _minmax_norm(title_scores)
        n_anchor = _minmax_norm(anchor_scores)
        n_pr = _minmax_norm(pr_scores)
        n_pv = _minmax_norm(pv_scores)

        # -------------------------
        # Combine weighted score
        # -------------------------
        final = defaultdict(float)
        for d in candidates:
            final[d] = (
                w["body"] * n_body.get(d, 0.0)
                + w["title"] * n_title.get(d, 0.0)
                + w["anchor"] * n_anchor.get(d, 0.0)
                + w["pagerank"] * n_pr.get(d, 0.0)
                + w["pageviews"] * n_pv.get(d, 0.0)
            )

        # Top-k selection (faster than sorting everything)
        top = heapq.nlargest(top_k, final.items(), key=lambda x: (x[1], -x[0]))

        # Attach titles
        return [(int(doc_id), self.meta.get_title(int(doc_id))) for doc_id, _ in top]