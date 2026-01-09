# title_module_test.py
import sys
from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[2]  # modules_checks -> src -> project root
# sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = Path(__file__).resolve().parents[1]  # modules_checks -> src
sys.path.insert(0, str(SRC_DIR))

import re
from title_module import TitleModule, TitleIndexConfig
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
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
    terms = tokenize(query)
    if not terms:
        return []

    config = TitleIndexConfig(
        base_dir="indexes/anchor_text",
        index_name="anchor_index",
        mode="gcs",
        bucket_name="ir_3_207472234",
        is_text_posting=False,
    )

    title_module = TitleModule(config)
    results = title_module.search(terms)

    return results


if __name__ == "__main__":
    q = "An American in Paris Basra"
    res = main(q)
    print(res[:100])  # show first 10 results
