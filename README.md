# Tern AI

A GPT-2-scale language model trained completely from scratch — random weight
initialization, no pretrained checkpoint — on a single consumer GPU, paired
with a self-built retrieval-augmented generation (RAG) system and live tool
use. Every component here, from the transformer to the vector database to
the retrieval pipeline, was built, measured, and debugged against real data
at real scale, not assumed to work from documentation.

## What this is

- **A 124M parameter GPT-2 (small)**, trained from random initialization on
  OpenWebText (~9B tokens, roughly one full epoch, in the same training
  regime as the original GPT-2's own published run), on a single RTX 4070
  SUPER (12GB VRAM). No cloud cluster, no pretrained weights borrowed.
- **A vector database built from scratch**: 6.4 million Wikipedia articles
  embedded and indexed for nearest-neighbor search, backed by FAISS
  (`IndexIVFFlat`) with cross-encoder reranking, not a naive brute-force
  cosine search — that approach was tried first, measured at ~7 seconds per
  query at this scale, and replaced once the numbers said so.
- **Retrieval-augmented generation + live tool use**: the fine-tuned model
  answers from retrieved Wikipedia context when it has one, calls a real
  calculator/datetime tool when the question calls for it, and says so
  honestly when it doesn't have a confident answer — with an optional live
  DuckDuckGo web search fallback that's explicitly kept out of the model's
  own training data (a rate-limited API can't be used to generate bulk
  training examples the way a one-time offline corpus download can).
- **A one-button training pipeline**: a single launcher detects which phase
  of training it's in — pretraining, fine-tuning, or done — and resumes or
  advances automatically, checkpointing every few minutes so a multi-day
  training run survives being paused, interrupted, or resumed on a
  different day.

## Why this project is worth a closer look

Small model, real engineering. The interesting part isn't the model size,
it's everything that had to be found, measured, and fixed to make the whole
pipeline actually work correctly on real hardware and real data, rather
than just running a tutorial:

- **Diagnosed a silent 4.6x training slowdown** (8s/iteration jumping to
  40s/iteration with no error, no crash) down to a Windows NVIDIA driver
  feature (CUDA sysmem fallback) silently spilling VRAM overflow into system
  RAM — invisible unless you go looking for it, since nothing reports it.
- **Found and fixed a real correctness bug** in the fine-tuning data itself:
  the model was being trained on the wrong shape of prompt for its own tool
  calling feature, meaning it would have silently never learned to call a
  tool at all — caught by actually running the trained model end-to-end
  against test cases, not by reading the code.
- **Tuned a real FAISS index against real recall failures**: a confirmed
  true top-3 nearest neighbor was missing from search results at the
  library's default settings; found by testing against the real 6.4 million
  row corpus, not a small sample, and fixed by measuring exactly how far a
  key parameter had to move before the miss actually went away.
- **A disambiguation-page filter grounded in Wikipedia's own style guide**
  rather than a guessed regex — and caught, via adversarial testing, deleting
  thousands of real articles by accident on its first attempt, then fixed
  again and verified with a real regression test suite before shipping.
- **A reranker upgrade that was tried, measured, and reverted** — the
  literature said it should help, real testing against the actual gate logic
  showed it broke a working case, so it didn't ship. Every change in this
  project is verified by running it, not by trusting a benchmark paper.

## Architecture

```
Wikipedia corpus ──▶ embed (sentence-transformers) ──▶ FAISS index ─┐
                                                                      ├──▶ retrieve() ──▶ chat loop ──▶ tagged, honest answer
User question ──▶ embed ──▶ cross-encoder rerank ────────────────────┘         │
                                                                                 ├─▶ CALL: tool(args) ──▶ real tool dispatch
                                                                                 └─▶ web: <question> ──▶ live DuckDuckGo search
```

The model is trained on three fixed prompt shapes (grounded context, tool
call, and honest "no match"), and the chat loop's job is choosing the right
one and grounding it in something real — the local index, a live tool, or a
live web search — rather than letting the model improvise from memory alone.

## Stack

PyTorch, a from-scratch GPT-2 implementation, `tiktoken`, FAISS,
`sentence-transformers` (embeddings + cross-encoder reranking), SQLite,
`ddgs` for live web search. No hosted training service, no managed vector
database, no third-party inference API for the model itself.
