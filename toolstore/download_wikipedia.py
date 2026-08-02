"""Download the Wikipedia summaries corpus (abokbot/wikipedia-first-paragraph)
into toolstore/corpus/wikipedia_summaries/ as plain text, one line per
article, title and summary separated by a tab. Resolves the "which source"
open question from SPEC.md in favor of abokbot over joejacobs: closer to
current Wikipedia, and loads with a single load_dataset() call, no custom
XML or wikitext parsing.

Schema confirmed directly against the dataset's own HuggingFace API listing:
columns are id, url, title, text (all string), single split named "train",
6,458,670 rows. text is the article's first paragraph only (short), built
upstream via text.split("\\n\\n")[0], not a full article.
"""
import os
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus", "wikipedia_summaries")
OUT_FILE = os.path.join(OUT_DIR, "summaries.tsv")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("abokbot/wikipedia-first-paragraph", split="train")
    print(f"rows: {len(ds)}")
    print(f"columns: {ds.column_names}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            title = row["title"].replace("\t", " ").replace("\n", " ").strip()
            text = row["text"].replace("\t", " ").replace("\n", " ").strip()
            if not title or not text:
                continue
            f.write(f"{title}\t{text}\n")

    print(f"wrote {OUT_FILE}")
