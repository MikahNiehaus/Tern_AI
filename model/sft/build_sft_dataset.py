"""Builds the SFT dataset from the three fixed template shapes (SPEC.md
Part 3), tokenized and loss masked, ready for train_sft.py's get_batch to
sample rows from directly.

Question synthesis is a fixed template, not real question generation
(researched and decided earlier: real QA generation needs an LLM call per
example, not viable for free at millions of examples). Tool shape reuses
the real, already executed call traces from toolstore/corpus/tool_examples,
not fabricated. No match shape is a small fixed set of greeting/trivial
inputs paired with plain conversational answers, so the model has seen the
pattern before being asked to produce it.

Tokenization and masking follow a real, standard technique (confirmed
against a real production fine tuning script, tloen/alpaca-lora/finetune.py):
never map a character index into a token index, tokenize the prompt only
prefix separately, its length in tokens is the real split point, since BPE
tokenization is deterministic left to right. Padded to block_size with
gpt2's own EOT token (50256, unused mid sequence, safe as a pad sentinel)
rather than packed, packing would need a document boundary attention mask
nanoGPT's model.py does not have, exactly the hand rolling this project
avoids.
"""
import json
import os

import numpy as np
import tiktoken

HERE = os.path.dirname(__file__)
TOOLSTORE = os.path.join(HERE, "..", "..", "toolstore")
WIKI_TSV = os.path.join(TOOLSTORE, "corpus", "wikipedia_summaries", "summaries.tsv")
TOOL_JSONL = os.path.join(TOOLSTORE, "corpus", "tool_examples", "calls.jsonl")
OUT_DIR = HERE

BLOCK_SIZE = 1024
PAD_ID = 50256  # gpt2 <|endoftext|>, unused mid sequence, safe as pad

# tens of thousands per shape, not millions: a "short" SFT stage per the
# schedule, not a second pretraining run. Real repetition count for
# template locking on a 124M model has no sourced magic number (LIMA's
# ~1000 examples is for a 65B model already carrying world knowledge, does
# not transfer down), so this is a reasoned starting point, not a citation,
# flagged as such. Roughly matched across shapes so the model does not
# learn to favor one shape over the others.
N_RAG = 30_000
N_TOOL_REPEATS = 1875  # 16 real traces * 1875 ~= 30,000
N_NOMATCH_REPEATS = 750  # (20 greeting + 20 factual refusal) * 750 ~= 30,000

NOMATCH_EXAMPLES = [
    ("hi", "Hello! What would you like to know?"),
    ("hello", "Hi there! How can I help?"),
    ("hey", "Hello! What can I do for you?"),
    ("thanks", "You're welcome!"),
    ("thank you", "You're welcome, happy to help."),
    ("ok", "Alright, let me know if you need anything else."),
    ("lol", "Glad that amused you!"),
    ("bye", "Goodbye! Have a great day."),
    ("goodbye", "Take care!"),
    ("how are you", "I'm doing well, thanks for asking! How can I help you today?"),
    ("what's up", "Not much, just here to help. What do you need?"),
    ("yo", "Hey! What can I help you with?"),
    ("sup", "Hey there! What's on your mind?"),
    ("cool", "Glad you think so!"),
    ("nice", "Thanks!"),
    ("haha", "Glad you found that funny!"),
    ("test", "I'm working! What would you like to ask?"),
    ("asdf", "I didn't quite understand that. Could you ask a real question?"),
    ("good morning", "Good morning! How can I help you today?"),
    ("good night", "Good night! Talk to you later."),
]

# Real factual questions, no fabricated ones, that this project's own
# retrieval testing confirmed find no usable match against the real store
# (workspace/nanogpt-from-scratch/context.md problem 20), plus more of the
# same shape (How/What/Who/When/Where phrasings across varied topics) so
# the model learns the general pattern, not 3 memorized topics. Before this,
# Context: (none) was only ever paired with a greeting, so a real factual
# question that finds no match put the model in a combination its SFT
# training never demonstrated. One fixed, short, consistent refusal
# sentence for all of them, not a varied one per example: matches the real
# published precedent for this (R-Tuning, arXiv:2311.09677, appends a
# short, consistent certainty marker rather than an elaborate one), and
# matches this project's own existing NOMATCH_EXAMPLES style already.
FACTUAL_NOMATCH_EXAMPLES = [
    ("How big is the Pacific Ocean?", "I don't have information about that."),
    ("How does the internet work?", "I don't have information about that."),
    ("What is the currency of the United Kingdom?", "I don't have information about that."),
    ("Who invented the telephone?", "I don't have information about that."),
    ("What is the boiling point of mercury?", "I don't have information about that."),
    ("Where is the deepest point in the ocean?", "I don't have information about that."),
    ("When was the printing press invented?", "I don't have information about that."),
    ("Why is the sky blue?", "I don't have information about that."),
    ("What is the tallest waterfall in the world?", "I don't have information about that."),
    ("Who discovered penicillin?", "I don't have information about that."),
    ("How many bones are in the human body?", "I don't have information about that."),
    ("What is the smallest country in the world?", "I don't have information about that."),
    ("When did the Berlin Wall fall?", "I don't have information about that."),
    ("What is the freezing point of nitrogen?", "I don't have information about that."),
    ("Who composed the Moonlight Sonata?", "I don't have information about that."),
    ("How deep is the Grand Canyon?", "I don't have information about that."),
    ("What causes the northern lights?", "I don't have information about that."),
    ("Where was the first Olympic Games held?", "I don't have information about that."),
    ("What is the largest desert in the world?", "I don't have information about that."),
    ("Who was the first person in space?", "I don't have information about that."),
]

enc = tiktoken.get_encoding("gpt2")


def tokenize_example(full_text, prefix_text):
    """Returns (x, y) each length BLOCK_SIZE. x is the full example padded
    with PAD_ID, y is x shifted by one for next token prediction, with -1
    (nanoGPT's own ignore_index) at every prompt and pad position.
    """
    full_ids = enc.encode_ordinary(full_text)
    prefix_ids = enc.encode_ordinary(prefix_text)
    split = len(prefix_ids)

    if len(full_ids) > BLOCK_SIZE:
        full_ids = full_ids[:BLOCK_SIZE]
        split = min(split, len(full_ids))

    # prefix tokens must match the full sequence's leading tokens exactly,
    # otherwise BPE fused a boundary character across the split (found the
    # hard way: a trailing space on the prefix fuses into the answer's
    # first word, e.g. "Answer: " + "Paris" tokenizes as "Answer:" + "
    # Paris", not "Answer:" + " " + "Paris"). Templates are built to end
    # right at "Answer:" with no trailing space specifically to avoid this,
    # but real data (titles, summaries) can still hit an edge case, back
    # off the split by one token rather than crash the whole build over one
    # example.
    while split > 0 and full_ids[:split] != prefix_ids[:split]:
        split -= 1

    labels = [-1] * split + full_ids[split:]
    pad_len = BLOCK_SIZE - len(full_ids)

    x = full_ids + [PAD_ID] * pad_len
    y = labels[1:] + [-1] * (pad_len + 1)
    return x, y


def rag_examples(limit):
    stride = None
    with open(WIKI_TSV, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    stride = max(1, total // limit)

    rows = []
    with open(WIKI_TSV, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride != 0:
                continue
            if len(rows) >= limit:
                break
            title, summary = line.rstrip("\n").split("\t", 1)
            question = f"What is {title}?"
            prefix = f"Context: {summary}\nQuestion: {question}\nAnswer:"
            full = prefix + " " + summary
            rows.append((full, prefix))
    return rows


def tool_examples(repeats):
    """Trains the model to GENERATE the CALL: line itself, not to be handed
    one. Bug 1 (found comparing this against SPEC.md's own Part 4 text,
    "the loop above trains the model to emit CALL: <tool_name>(<args>)"):
    the first version of this function put the call line in the masked
    prefix (given, not predicted) and only trained a natural language
    restatement of a precomputed result after it, so the model would never
    actually learn to produce CALL: itself. Fixed by moving CALL: into the
    trained continuation. Bug 2 (found by bad-cop, real execution, not
    review): that fix still used `f"Question: {question}\nAnswer:"` as the
    prefix, but chat.py's build_prompt() never sends that shape, it always
    prepends a Context: line, real content or "(none)". Confirmed by
    decoding real chat.py behavior end to end: the two prefixes were
    structurally unrelated at token 0. Fixed here to match build_prompt()'s
    no-match branch exactly, "Context: (none)\nQuestion: ...\nAnswer:",
    since tool questions (arithmetic, current time) are not expected to
    clear the Wikipedia rerank threshold. Residual, accepted risk, same
    category SPEC.md Part 4 already documents for "hi"/"lol": a tool
    question that happens to coincidentally match a real Wikipedia article
    (confirmed for real: "What is 2 plus 2?" retrieved "Plus Two (film)" at
    score 8.55, above the 7.0 threshold) would build a Context-having
    prompt the model was not trained to answer with CALL: for. Not solved
    here, same "good enough to ship, not perfectly clean" tolerance already
    accepted elsewhere in this project for corpus-coincidence collisions.
    The precomputed result/ok fields from calls.jsonl are not needed here,
    chat.py runs the tool for real and gets a live result instead of
    training the model to guess or memorize one.
    """
    traces = []
    with open(TOOL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))

    rows = []
    for _ in range(repeats):
        for t in traces:
            question = t["question"]
            call = t["call"]
            prefix = f"Context: (none)\nQuestion: {question}\nAnswer:"
            full = prefix + " " + call
            rows.append((full, prefix))
    return rows


def nomatch_examples(repeats):
    rows = []
    all_examples = NOMATCH_EXAMPLES + FACTUAL_NOMATCH_EXAMPLES
    for _ in range(repeats):
        for question, answer in all_examples:
            prefix = f"Context: (none)\nQuestion: {question}\nAnswer:"
            full = prefix + " " + answer
            rows.append((full, prefix))
    return rows


def build():
    print("generating RAG shape examples from Wikipedia summaries...")
    rows = rag_examples(N_RAG)
    print(f"  {len(rows)} RAG examples")

    print("generating tool shape examples from real call traces...")
    tool_rows = tool_examples(N_TOOL_REPEATS)
    print(f"  {len(tool_rows)} tool examples")
    rows.extend(tool_rows)

    print("generating no match shape examples...")
    nomatch_rows = nomatch_examples(N_NOMATCH_REPEATS)
    print(f"  {len(nomatch_rows)} no match examples")
    rows.extend(nomatch_rows)

    print(f"tokenizing {len(rows)} total examples...")
    rng = np.random.default_rng(1337)
    rng.shuffle(rows)

    xs = np.zeros((len(rows), BLOCK_SIZE), dtype=np.uint16)
    ys = np.zeros((len(rows), BLOCK_SIZE), dtype=np.int32)  # int32: needs to hold -1
    for i, (full, prefix) in enumerate(rows):
        x, y = tokenize_example(full, prefix)
        xs[i] = x
        ys[i] = y
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(rows)} tokenized")

    n_val = max(1, len(rows) // 20)  # 5% val
    val_x, val_y = xs[:n_val], ys[:n_val]
    train_x, train_y = xs[n_val:], ys[n_val:]

    np.save(os.path.join(OUT_DIR, "sft_train_x.npy"), train_x)
    np.save(os.path.join(OUT_DIR, "sft_train_y.npy"), train_y)
    np.save(os.path.join(OUT_DIR, "sft_val_x.npy"), val_x)
    np.save(os.path.join(OUT_DIR, "sft_val_y.npy"), val_y)
    print(f"wrote {len(train_x)} train and {len(val_x)} val examples to {OUT_DIR}")


if __name__ == "__main__":
    build()
