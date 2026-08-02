"""One-off driver: re-embeds the full corpus with the new embedding model
(embed.py now uses BAAI/bge-small-en-v1.5) into a fresh vectorstore_v2.db,
then rebuilds vectorstore_v2.faiss from it. Builds into new v2 paths, not
in place, so the currently working vectorstore.db/.faiss stay untouched
and queryable until the new ones are verified and swapped in by hand.

Run with the Start-Process pattern from CLAUDE.md (this runs well past
the ~15 minute background-kill window a normal backgrounded call hits).
"""
import os

import build_faiss_index
import build_index

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "vectorstore_v2.db")
FAISS_PATH = os.path.join(HERE, "vectorstore_v2.faiss")

if __name__ == "__main__":
    print("=== step 1: re-embedding corpus with the new model into vectorstore_v2.db ===", flush=True)
    build_index.build(db_path=DB_PATH)
    print("=== step 2: building vectorstore_v2.faiss from the new embeddings ===", flush=True)
    build_faiss_index.build(db_path=DB_PATH, faiss_index_path=FAISS_PATH)
    print("=== done: vectorstore_v2.db and vectorstore_v2.faiss ready for verification ===", flush=True)
