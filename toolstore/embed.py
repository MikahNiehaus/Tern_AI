"""Turns raw corpus text into vectors with sentence-transformers, per
SPEC.md Part 2. Embeddings are normalized before serializing (normalize_embeddings=True)
so that plain L2 distance in the vec0 table ranks the same as cosine distance
would, matching the known failure mode already documented in SPEC.md: cosine
similarity is wrong if vectors are not normalized first.

Forced onto CPU explicitly. Found the hard way: sentence-transformers
auto-selects CUDA when available, and left implicit this silently competed
with GPU training for the same card, a real measured 40s/iteration slowdown
(4.6x) traced back to exactly this. The design was always "runs on CPU in
parallel with GPU training," this makes that the actual code, not just the
stated intent.
"""
from sentence_transformers import SentenceTransformer
from sqlite_vec import serialize_float32

_MODEL = None

# BAAI/bge-small-en-v1.5, swapped in for a real measured quality gap (this
# project's own retrieval testing found roughly half of a diverse 20
# question batch missed or landed on a tangential article). Same 384
# dimension as the model it replaces (confirmed against its own model
# card), so schema.sql's vec0(embedding float[384]) and
# build_faiss_index.py's DIM = 384 need no change, only a re embed and
# re index. 33.4M params vs the previous 22.7M, about 1.5x the compute,
# same order of magnitude as the real 38 minute uncontended / 16 hour
# contended numbers already measured for the full 6.4 million row corpus.
_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE v1.5's own model card recommends prefixing the QUERY text (never the
# corpus text) with this instruction for asymmetric retrieval, and states
# skipping it "has only a slight degradation" rather than being required.
# Applied only in embed_text() (query.py's only caller, one query per
# call), never in embed_texts() (build_index.py's only caller, corpus
# documents), since the two are already naturally separated by call site,
# no new parameter needed to keep them apart.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _MODEL


def embed_texts(texts):
    """texts: list[str]. Returns list[bytes], one serialized float32 blob per text."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [serialize_float32(v.tolist()) for v in vectors]


def embed_text(text):
    return embed_texts([_QUERY_INSTRUCTION + text])[0]
