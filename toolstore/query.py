"""Retrieval call for the chat loop (SPEC.md Part 4), corrected after
measuring real production scale: sqlite-vec's own brute force MATCH search
took about 7 seconds per query against the real 6.4 million row corpus
(sqlite-vec's own tracking issue confirms brute force only, recommended up
to a few hundred thousand rows). Replaced with FAISS IndexIVFFlat
(build_faiss_index.py) for the actual nearest neighbor search, sqlite still
holds metadata (title, text, type), looked up by id after FAISS returns
candidates.

A raw distance threshold was also found broken at real scale: "hi" and
"lol" landed close enough to real Wikipedia articles to beat some
genuinely relevant queries on distance alone. Replaced with reranking the
FAISS candidates against the actual query text using a small local cross
encoder (cross-encoder/ms-marco-MiniLM-L-6-v2), gating the no match
decision on that score instead of raw embedding distance.
"""
import os
import re
import sqlite3

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_text

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "vectorstore.db")
FAISS_INDEX_PATH = os.path.join(HERE, "vectorstore.faiss")

_FAISS_INDEX = None
_RERANKER = None

# Wikipedia disambiguation pages, not full articles, still real rows in this
# title+first-paragraph-only dataset (abokbot/wikipedia-first-paragraph has
# no disambiguation flag, confirmed by reading its own dataset card, and the
# live MediaWiki API metadata that would normally flag one, confirmed by
# reading the real `wikipedia` package's own source, isn't available
# offline against already embedded rows). Grounded in Wikipedia's own documented
# style rules instead (Wikipedia:Manual of Style/Disambiguation pages,
# MOS:DAB), the two conventions that survive into title + first paragraph
# alone: the "(disambiguation)" title suffix, and MOS:DAB's introductory
# line, which names the term, says "refer(s) to", and ends in a colon
# introducing the list of meanings.
#
# All three parts of the lead pattern are load bearing, each one measured
# against the real 6.4 million row store, not assumed:
#
# 1. Anchored to the first 80 characters, not searched unbounded. An
#    unbounded search false matched real substantive articles using "may
#    refer to" mid paragraph as ordinary prose (rowid 1194 "Ackermann
#    function", rowid 3162 "Cranberry", rowid 264 "Adelaide").
# 2. "refers? to", not the literal "may (also) refer to". Real rows use
#    every variant: rowid 1374 "BLM" ("most commonly refers to:"), rowid
#    4243 "English" ("usually refers to:"), rowid 688 "Auriga" ("can refer
#    to:"), rowid 24 "Alien" ("primarily refers to:").
# 3. The trailing colon, which is what separates a disambiguation stub from
#    an ordinary definition, and is required. "X refers to Y" with no colon
#    is the standard encyclopedic opening of a real article, not a stub.
#    Dropping the colon requirement deleted rowid 18735 "Inflation", rowid
#    23979 "Software architecture", "Cryptanalysis", "Middle East",
#    "Freemasonry" and 0.33% of the whole corpus (~21,000 real articles,
#    measured over a 178,424 row sample spanning every rowid range) from
#    every future query, with no way to retrieve them again.
#
# The colon requirement does miss the minority of stubs whose flattened
# first paragraph lost its colon ("Bristol buses may refer to  Bristol, the
# make of bus..."). That direction is the cheap one to be wrong in: a missed
# stub costs one weak Context on one query, a wrongly deleted article is
# unreachable forever. The `\s*$` alternative catches the content free
# stubs, where "refer to" is the last thing in the row (rowid 4182
# "Discharge", rowid 22235 "Hackney").
_DAB_TITLE_RE = re.compile(r"\(disambiguation\)\s*$", re.IGNORECASE)
_DAB_LEAD_RE = re.compile(r"^.{0,80}\brefers? to\b(?:[^:\n]{0,60}:|\s*$)", re.IGNORECASE)


def _is_disambiguation(title, content):
    return bool(_DAB_TITLE_RE.search(title or "")) or bool(_DAB_LEAD_RE.search(content or ""))


def connect(db_path=DB_PATH):
    return sqlite3.connect(db_path)


def get_faiss_index(faiss_index_path=FAISS_INDEX_PATH, nprobe=512):
    # 32 was the original default (an nlist=65536 IVF index searches
    # nprobe/nlist of its clusters, 32 meant about 0.05%), only ever load
    # tested (SPEC.md's ~0.02-0.03s number), never recall tested at the real
    # 6.4 million row scale. Real bug found by testing it: "What is the
    # Sun?" never surfaced the real, existing "Sun" row anywhere in the top
    # 20 FAISS candidates at nprobe=32, recovered by 128. "Elon Musk" needed
    # more: confirmed a true rank 3 nearest neighbor with an exhaustive
    # search (nprobe=nlist, FAISS's own wiki confirms this is mathematically
    # equivalent to brute force), so this was a real IVF routing gap, not a
    # ranking limit, and 128 still missed it. Swept the real range: recovers
    # at 512 (rank 2), no further gain past that. Cost still small relative
    # to the reranker and model.generate(), measured 27ms at 512 versus
    # 11.3ms at 32, for a single local interactive user.
    global _FAISS_INDEX
    if _FAISS_INDEX is None:
        _FAISS_INDEX = faiss.read_index(faiss_index_path)
        _FAISS_INDEX.nprobe = nprobe
    return _FAISS_INDEX


def get_reranker():
    # forced onto CPU explicitly, same reason as embed.py: left implicit,
    # this silently competes with GPU training for the same card, a real
    # measured 40s/iteration (4.6x) slowdown traced back to exactly this
    #
    # BAAI/bge-reranker-base was tried here, on real, grounded reasoning (a
    # real ablation study, arXiv 2409.07691, found reranker quality is what
    # actually fixes "correct article ranked below a plausible but wrong
    # one" failures) and reverted after real testing contradicted it: with
    # Sigmoid activation its scores saturated near 1.0 for everything
    # including "hi" (0.9998) and "thanks" (0.9993), no usable separation
    # left for the threshold gate. Without activation_fn its raw scores
    # still put "hi" at 1.000, identical to genuinely correct matches
    # ("What is the Sun?" also 1.000), which would have silently broken the
    # reject gate this file's own docstring already documents fighting to
    # get working once. The theoretical case for this swap was real, the
    # measured behavior against this project's own threshold based design
    # was not, so it was not kept, matching this project's own standing
    # rule to verify by running, not by reasoning.
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cpu")
    return _RERANKER


def search(db, query_text, k=20, final_k=5, rerank_threshold=None, type_filter=None):
    """Returns a list of dicts: rowid, rerank_score, type, title, content,
    sorted best first. Empty list if nothing clears rerank_threshold (the
    "hi" case from SPEC.md Part 4, not an error, just no usable match).

    k stays at 20. Raising it to 50 was tried, to make "What is DNA?" and
    "What is the capital of France?" return something instead of no match,
    and reverted, because measuring what k=50 actually returns shows it
    buys nothing and costs the rerank gate:

    1. It breaks the gate. Found by bad-cop, real execution: at k=25 and
       above, "hi" clears the 7.0 threshold against a real but
       coincidentally titled article ("Hi (magazine)", 8.85), the exact
       false match the cross encoder rerank exists to prevent (see this
       file's own top docstring). At k=20 "hi" correctly returns nothing.
    2. What it "recovers" is not the article anyone wanted. Re-measured
       after fixing the disambiguation filter: k=50 returns "DNA (1997
       film)" at 7.83 for "What is DNA?" and "Administration of Paris" at
       8.10 for "What is the capital of France?", both coincidental
       neighbours scraping over the threshold, not the real DNA or France
       rows. Grounding an answer in those is worse than not grounding it.

    So this is not a tradeoff between two good outcomes, and raising k is
    not worth retrying. No match is also the well trained outcome here:
    model/sft/build_sft_dataset.py's FACTUAL_NOMATCH_EXAMPLES trains real
    factual questions paired with Context: (none) and a short refusal, and
    its NOMATCH_EXAMPLES greetings are only ever paired with
    Context: (none) too (verified by decoding the real generated rows), so
    "hi" retrieving an article would hand the model a prompt shape it has
    never seen in training.
    """
    index = get_faiss_index()
    query_vec = np.frombuffer(embed_text(query_text), dtype=np.float32).reshape(1, -1)
    _, ids = index.search(query_vec, k)
    ids = [int(i) for i in ids[0] if i != -1]
    if not ids:
        return []

    placeholders = ",".join("?" * len(ids))
    sql = f"SELECT rowid, type, title, content FROM metadata WHERE rowid IN ({placeholders})"
    params = list(ids)
    if type_filter is not None:
        sql += " AND type = ?"
        params.append(type_filter)
    rows = db.execute(sql, params).fetchall()
    candidates = [{"rowid": r[0], "type": r[1], "title": r[2], "content": r[3]} for r in rows]
    candidates = [c for c in candidates if not _is_disambiguation(c["title"], c["content"])]
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query_text, c["content"]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    if rerank_threshold is not None:
        candidates = [c for c in candidates if c["rerank_score"] >= rerank_threshold]
    return candidates[:final_k]


def retrieve(query_text, k=20, final_k=5, rerank_threshold=7.0):
    """Adapter conforming to retrieval_types.Retriever, so chat.py depends
    on this shape, not on search()'s FAISS/sqlite specific parameters
    (db handle, rowid, k vs final_k). Swapping in a different backend later
    (a web search API, a different vector store) means writing a new module
    with this same retrieve(query_text) -> list[RetrievedDoc] signature,
    chat.py would not need to change. Everything FAISS/sqlite/cross encoder
    specific stays inside this file, behind this one function.

    type_filter="fact" always. build_index.py also writes the 16 real tool
    call traces into this same store as type='tool' rows (SPEC.md Part 2's
    one shared store for both corpora); nothing reads them back out, the
    SFT builder loads calls.jsonl from disk instead
    (model/sft/build_sft_dataset.py::tool_examples), so they are index
    ballast here, never retrieval results. Unfiltered they were worse than
    ballast: measured against the real store, 11 of the 16 tool questions
    retrieved their own near-identical trace (e.g. "What is 17 times 23?"
    at rerank score 7.95, above the 7.0 gate), which would build a
    Context:-having RAG prompt for exactly the questions the model is
    trained to answer with CALL: from Context: (none). Post-filtering
    rather than pre-filtering costs recall in proportion to how much of the
    corpus it excludes; here that is 16 rows out of 6.4 million, so the
    cheap option is the right one (the Faiss paper, arXiv 2401.08281,
    names post- and pre-filtering as the two standard approaches).
    """
    db = connect()
    try:
        results = search(db, query_text, k=k, final_k=final_k, rerank_threshold=rerank_threshold, type_filter="fact")
    finally:
        db.close()
    return [
        {
            "content": r["content"],
            "metadata": {"rowid": r["rowid"], "type": r["type"], "title": r["title"]},
            "score": r["rerank_score"],
        }
        for r in results
    ]
