"""Migrates the embeddings already computed and stored in sqlite-vec into a
FAISS IndexIVFFlat, per SPEC.md Part 2's correction: sqlite-vec's own
tracking issue (asg017/sqlite-vec#25) confirms brute force only in the
stable release, recommended up to a few hundred thousand vectors, this
corpus is 6.4 million, measured directly at about 7 seconds per query.

Reads the embeddings back out of vec_items losslessly (float32, no
re-embedding needed, that was the expensive part) via numpy.frombuffer,
which round trips exactly with sqlite_vec.serialize_float32 (verified).
METRIC_INNER_PRODUCT because embed.py already normalizes vectors, so inner
product ranks the same as cosine distance, and IndexFlatIP is the matching
quantizer, confirmed against FAISS's own MetricType docs, not just assumed
by analogy to the L2 tutorial case. add_with_ids is natively supported on
IndexIVFFlat (confirmed against FAISS's own C++ API docs, no IndexIDMap
wrapper needed), keeping the existing rowid to metadata mapping intact.

nlist is not the small dataset 4*sqrt(N) formula, confirmed against FAISS's
own wiki (Guidelines to choose an index): that formula is for under 1M
vectors, this corpus is 6.4 million, in the 1M to 10M tier where the wiki
recommends nlist=65536 directly. Sticking with a flat (IndexFlatIP)
quantizer rather than the wiki's HNSW32 quantizer suggestion for that tier,
simpler to build correctly, the coarse quantizer search cost at 65536
centroids is still small next to the corpus itself.
"""
import argparse
import os
import time

import faiss
import numpy as np
import sqlite3
import sqlite_vec

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "vectorstore.db")
FAISS_INDEX_PATH = os.path.join(HERE, "vectorstore.faiss")

DIM = 384
NLIST = 65536  # FAISS wiki's own recommendation for the 1M-10M vector tier
BATCH_SIZE = 50_000


def connect(db_path):
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def build(db_path=DB_PATH, faiss_index_path=FAISS_INDEX_PATH, nlist=NLIST):
    db = connect(db_path)
    total = db.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
    print(f"{total} vectors to migrate into FAISS, nlist={nlist}")

    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)

    # FAISS wiki: 30x to 256x nlist training vectors, the more the better,
    # 40x here, well inside that band and comfortably under the total corpus size
    train_n = min(total, nlist * 40)
    print(f"training on {train_n} sample vectors")
    # ORDER BY random() over 6.4M rows is a known SQLite anti-pattern, a full
    # table scan plus sort, found slow the hard way (still running after 8
    # minutes with no output). Stride sampling by rowid instead: no sort
    # needed, still spreads evenly across the whole corpus.
    stride = max(1, total // train_n)
    train_rows = db.execute(
        "SELECT rowid, embedding FROM vec_items WHERE rowid % ? = 0 LIMIT ?", [stride, train_n]
    ).fetchall()
    train_vecs = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in train_rows])
    t0 = time.time()
    index.train(train_vecs)
    print(f"trained in {time.time() - t0:.1f}s")

    last_rowid = 0
    added = 0
    t0 = time.time()
    while True:
        try:
            rows = db.execute(
                "SELECT rowid, embedding FROM vec_items WHERE rowid > ? ORDER BY rowid LIMIT ?",
                [last_rowid, BATCH_SIZE],
            ).fetchall()
            if not rows:
                print(f"query returned no more rows after last_rowid={last_rowid}, stopping", flush=True)
                break
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            index.add_with_ids(vecs, ids)
            added += len(rows)
            # found the hard way, confirmed against python/cpython#106878:
            # sqlite3 does not recognize numpy integer types as valid
            # numeric parameters, binds them as raw bytes instead, and
            # silently matches nothing rather than raising. int() is the
            # documented workaround.
            last_rowid = int(ids[-1])
            elapsed = time.time() - t0
            print(f"{added}/{total} added to FAISS, {added/elapsed:.1f} vecs/sec, last_rowid={last_rowid}", flush=True)
        except Exception as e:
            print(f"EXCEPTION at added={added}, last_rowid={last_rowid}: {type(e).__name__}: {e}", flush=True)
            raise

    faiss.write_index(index, faiss_index_path)
    print(f"done: wrote {faiss_index_path}, {index.ntotal} vectors", flush=True)
    db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--faiss-index-path", default=FAISS_INDEX_PATH)
    a = p.parse_args()
    build(db_path=a.db_path, faiss_index_path=a.faiss_index_path)
