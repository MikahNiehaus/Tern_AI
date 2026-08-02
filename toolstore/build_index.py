"""Writes fact (Wikipedia summaries) and tool rows into the shared sqlite-vec
store, per SPEC.md Part 2. Connection/load/insert pattern taken directly
from sqlite-vec's own examples/simple-python/demo.py.

--limit N processes only the first N fact rows, for measuring real embedding
throughput on this machine before committing to the full ~6.5 million row
pass (SPEC.md's own open question: no sourced throughput number exists for
all-MiniLM-L6-v2, needs a real local timing test).
"""
import argparse
import json
import os
import sqlite3
import time

import sqlite_vec

from embed import embed_texts

HERE = os.path.dirname(__file__)
DEFAULT_DB_PATH = os.path.join(HERE, "vectorstore.db")
SCHEMA_PATH = os.path.join(HERE, "schema.sql")
DEFAULT_WIKI_TSV = os.path.join(HERE, "corpus", "wikipedia_summaries", "summaries.tsv")
TOOL_JSONL = os.path.join(HERE, "corpus", "tool_examples", "calls.jsonl")

BATCH_SIZE = 256


def connect(db_path):
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        db.executescript(f.read())
    return db


def insert_batch(db, rows):
    """rows: list of (rowid, type, title, content)."""
    texts = [r[3] for r in rows]
    blobs = embed_texts(texts)
    with db:
        for (rowid, type_, title, content), blob in zip(rows, blobs):
            db.execute(
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                [rowid, blob],
            )
            db.execute(
                "INSERT INTO metadata(rowid, type, title, content) VALUES (?, ?, ?, ?)",
                [rowid, type_, title, content],
            )


def iter_wiki_rows(wiki_tsv, limit=None):
    with open(wiki_tsv, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            title, text = line.rstrip("\n").split("\t", 1)
            yield ("fact", title, text)


def iter_tool_rows():
    with open(TOOL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            content = f"{row['question']} {row['call']} -> {row['result']}"
            yield ("tool", row["call"], content)


def build(limit=None, include_tools=True, db_path=DEFAULT_DB_PATH, wiki_tsv=DEFAULT_WIKI_TSV, resume=False):
    db = connect(db_path)

    already_fact = 0
    already_tool = 0
    if resume:
        already_fact = db.execute("SELECT COUNT(*) FROM metadata WHERE type = 'fact'").fetchone()[0]
        already_tool = db.execute("SELECT COUNT(*) FROM metadata WHERE type = 'tool'").fetchone()[0]
        if already_fact or already_tool:
            print(f"resuming: {already_fact} fact rows and {already_tool} tool rows already in {db_path}")

    rowid = already_fact + already_tool + 1
    t0 = time.time()
    batch = []
    total = 0

    def flush():
        nonlocal batch, total
        if batch:
            insert_batch(db, batch)
            total += len(batch)
            batch = []

    skipped = 0
    for type_, title, content in iter_wiki_rows(wiki_tsv, limit=limit):
        if skipped < already_fact:
            skipped += 1
            continue
        batch.append((rowid, type_, title, content))
        rowid += 1
        if len(batch) >= BATCH_SIZE:
            flush()
            elapsed = time.time() - t0
            print(f"{total} rows embedded this run, {total/elapsed:.1f} rows/sec")
    flush()

    if include_tools and already_tool == 0:
        for type_, title, content in iter_tool_rows():
            batch.append((rowid, type_, title, content))
            rowid += 1
        flush()

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0.0
    print(f"done: {total} rows embedded this run in {elapsed:.1f}s, {rate:.1f} rows/sec")
    db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only process first N wikipedia rows, for throughput testing")
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--wiki-file", default=DEFAULT_WIKI_TSV)
    p.add_argument("--resume", action="store_true", help="skip rows already present in db_path instead of starting over")
    a = p.parse_args()
    build(limit=a.limit, include_tools=not a.no_tools, db_path=a.db_path, wiki_tsv=a.wiki_file, resume=a.resume)
