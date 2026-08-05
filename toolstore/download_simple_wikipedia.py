"""Download the Simple English Wikipedia corpus into
toolstore/corpus/simple_wikipedia_summaries/ as plain text, one line per
article, title and first paragraph separated by a tab. Same shape as
download_wikipedia.py's own output, a separate file, not merged with it:
build_sft_dataset.py joins the two by title at SFT build time, it does not
need them pre merged.

No prebuilt "Simple Wikipedia, first paragraph only" dataset exists on
HuggingFace (checked directly against the hub's own search API, abokbot
never published a Simple English sibling to wikipedia-first-paragraph), and
the legacy `load_dataset("wikipedia", "20220301.simple")` script loader is
documented broken in practice (huggingface/datasets#4327, "takes hours and
gets killed"). wikimedia/wikipedia's own "20231101.simple" config is the
real, current, working source: confirmed directly against the dataset's own
row API, same four columns as abokbot's dataset (id, url, title, text), but
text is the FULL article here, not already split to just the first
paragraph the way abokbot's upstream build already was, hence the one extra
split step below that download_wikipedia.py did not need.
"""
import os
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus", "simple_wikipedia_summaries")
OUT_FILE = os.path.join(OUT_DIR, "summaries.tsv")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    print(f"rows: {len(ds)}")
    print(f"columns: {ds.column_names}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            title = row["title"].replace("\t", " ").replace("\n", " ").strip()
            # Paragraphs are separated by a blank line, the same convention
            # abokbot's own upstream build used for the English corpus
            # (confirmed against download_wikipedia.py's own docstring),
            # just not split for us ahead of time this time.
            first_para = row["text"].split("\n\n")[0]
            text = first_para.replace("\t", " ").replace("\n", " ").strip()
            if not title or not text:
                continue
            f.write(f"{title}\t{text}\n")

    print(f"wrote {OUT_FILE}")
