# Tern AI — 3.5 Minute Technical Walkthrough (script)

Target: ~500-540 words, ~3.5 minutes at a normal speaking pace. Read it once out loud before recording; trim anything that feels rushed rather than speeding up delivery.

---

**[0:00 - 0:25] The constraint**

This is a 124 million parameter language model, trained from random initialization, on a single consumer GPU. No pretrained weights, no cluster. The entire architecture and training schedule had to fit inside 12 gigabytes of VRAM, and that constraint drove almost every engineering decision in this project.

**[0:25 - 0:55] Architecture and pretraining**

The model itself is a standard GPT-2 small: 12 transformer layers, 12 attention heads, 768 dimensions, 123.6 million parameters. It trained on OpenWebText, about 9 billion tokens, for 25,000 iterations with mixed precision and a cosine learning rate schedule. The result: a validation loss of 3.07. For comparison, OpenAI's own original GPT-2 checkpoint scores 3.11 on that same dataset. This model, trained from scratch on one GPU, already matches that baseline.

**[0:55 - 1:35] The batch size that lied**

One of the real engineering problems: an isolated benchmark said batch size 8 fit comfortably in memory. Running the actual training loop told a different story: 39 seconds per iteration, under 3 percent of the GPU's real throughput. The cause was a documented NVIDIA driver behavior: when VRAM gets nearly full, allocations silently spill into system RAM instead of raising an out-of-memory error. Nothing crashes, it just gets five times slower with zero warning. Dropping to batch size 6 and adjusting gradient accumulation to keep the same effective batch fixed it: 8 seconds per iteration, five times faster, no functional change to training dynamics.

**[1:35 - 2:05] Retrieval, and a bug that hid in plain sight**

On top of the language model sits a retrieval system: 6.4 million Wikipedia articles, embedded and indexed with FAISS for fast approximate search, then reranked with a cross-encoder so a coincidental nearest neighbor never gets treated as a real answer. At one point, factual questions started returning completely unrelated articles. The cause: the corpus had been indexed with one embedding model, while live queries were using an upgraded one. Two different embedding models place identical text in different coordinate systems, so every query was comparing against noise. Re-embedding the full corpus with a consistent model fixed it, verified against the exact query that had been failing.

**[2:05 - 2:40] Fine tuning, and never letting it copy**

Fine tuning teaches the model its output format, not new facts — supervised fine tuning is mostly about behavior, not knowledge. It learns three fixed shapes: paraphrase a retrieved passage, emit a structured call to a real tool like a calculator, or say it doesn't know. The retrieval-grounded shape is trained exclusively on real, independently written paraphrases, never the source text copied verbatim — the fine tuning dataset even records the exact set of article titles it actually learned to paraphrase, and the chat loop checks every retrieval hit against that list before ever answering from it, so a title it wasn't trained on falls back to an honest "I don't know" instead of a copy-paste.

**[2:40 - 3:05] Always generating, never just printing**

Tool results are the one deliberate exception: a calculator's answer is already correct, so it prints directly rather than risking a second pass introducing an error. Everything else — a grounded answer, or a live web search when the model doesn't know — is always the model actually generating from what it was shown, never text handed back unchanged. Best-of-4 sampling generates several real candidate answers and keeps the one a reranker scores as most relevant, and nucleus sampling narrows word choice at every token, both real, standard inference-time techniques, no retraining required for either.

**[3:05 - 3:20] Close**

Every part of this, from the driver level VRAM bug to the embedding mismatch to the model's own training data, was found and fixed by measuring the actual system, not by guessing. The finished checkpoint is public on Hugging Face for anyone who wants to skip the training run and just talk to it.
