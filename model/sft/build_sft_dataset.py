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
import hashlib
import json
import os
import re
import sys

import numpy as np
import tiktoken

HERE = os.path.dirname(__file__)
TOOLSTORE = os.path.join(HERE, "..", "..", "toolstore")
WIKI_TSV = os.path.join(TOOLSTORE, "corpus", "wikipedia_summaries", "summaries.tsv")
SIMPLE_WIKI_TSV = os.path.join(TOOLSTORE, "corpus", "simple_wikipedia_summaries", "summaries.tsv")
TOOL_JSONL = os.path.join(TOOLSTORE, "corpus", "tool_examples", "calls.jsonl")
OUT_DIR = HERE

# Same real, already tested first_sentence() toolstore/chat.py uses to cut a
# generated answer down to one sentence at inference time (not reimplemented
# here, not the naive regex two earlier rounds of this file already proved
# wrong): training the paraphrase target itself as one concise sentence,
# not chat.py's max_new_tokens budget plus a post hoc cut, so a short answer
# is what the model actually learned to produce, not a side effect of
# truncating a longer one after the fact.
sys.path.insert(0, TOOLSTORE)
from sentence_boundary import first_sentence

# Copied from toolstore/query.py's own _is_disambiguation() (its docstring
# there documents three load reasons for the exact pattern shape: the 80
# char anchor, the "refers? to" variants, the trailing colon requirement),
# not imported: query.py also imports faiss and sentence_transformers for
# its own live retrieval path, multiple seconds of import cost this one time
# data build script has no other reason to pay. Used here to keep
# disambiguation stub pages out of the Simple Wikipedia paraphrase source,
# the same reason query.py keeps them out of retrieval results, a thin stub
# would make a false, information free "paraphrase."
_DAB_TITLE_RE = re.compile(r"\(disambiguation\)\s*$", re.IGNORECASE)
_DAB_LEAD_RE = re.compile(r"^.{0,80}\brefers? to\b(?:[^:\n]{0,60}:|\s*$)", re.IGNORECASE)


def _is_disambiguation(title, content):
    return bool(_DAB_TITLE_RE.search(title or "")) or bool(_DAB_LEAD_RE.search(content or ""))


MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "dataset_manifest.json")
# The exact set of titles rag_examples() actually turned into a training
# row, written by build() and read by toolstore/chat.py's
# _get_paraphrase_titles(): the single source of truth for "did the model
# really see a trained paraphrase for this title," not a re-derived
# approximation. rag_examples() streams WIKI_TSV in file order and stops at
# N_RAG matches, so the real trained set is a small, order-dependent subset
# of every title load_simple_wikipedia_paraphrases() could theoretically
# supply a paraphrase for (bad-cop measured a real 178,363-title, 74.8%
# mismatch between the two before this file existed: chat.py's gate was
# treating titles as trained that rag_examples() never actually sampled).
TRAINED_RAG_TITLES_PATH = os.path.join(os.path.dirname(__file__), "trained_rag_titles.json")


def _file_signature(path):
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def _input_fingerprint():
    """A cheap, size and modified time based fingerprint of everything the
    SFT dataset build actually reads: both Wikipedia corpora, the tool call
    traces, and this script's own source, so a real logic change here (like
    the paraphrase feature this fingerprint was added alongside) counts as
    a real input change too, the same as the corpus files changing.

    Hashing the full content of a 6.4 million row corpus on every
    orchestrate.py run to answer "did this change" would be needlessly
    slow for a check meant to run before every single resume decision; size
    and modified time is the same cheap proxy real build systems (make,
    ccache) already use for this exact question, and it is what decides
    whether dataset_needs_rebuild() below tells orchestrate.py the existing
    sft_train_x.npy etc. are still trustworthy.

    Includes sentence_boundary.py's own hash, not just this script's:
    load_simple_wikipedia_paraphrases() calls its first_sentence() to build
    the exact text baked into every RAG training row, so a real change
    there (bad-cop found this file has five real prior rounds of lexicon
    and regex changes) changes the dataset just as much as an edit to this
    script does, and must count as a real input change the same way.
    """
    with open(__file__, "rb") as f:
        script_hash = hashlib.sha256(f.read()).hexdigest()
    return {
        "wiki_tsv": _file_signature(WIKI_TSV),
        "simple_wiki_tsv": _file_signature(SIMPLE_WIKI_TSV),
        "tool_jsonl": _file_signature(TOOL_JSONL),
        "build_script_sha256": script_hash,
        "sentence_boundary_sig": _file_signature(os.path.join(TOOLSTORE, "sentence_boundary.py")),
    }


def dataset_needs_rebuild():
    """True if the tokenized .npy files are missing, or the real inputs
    that built them (either Wikipedia corpus, the tool traces, or this
    script's own logic) have changed since the last real build.

    Exists so orchestrate.py's resume check is never just "do these files
    exist," which would silently keep training (or silently launch chat)
    on a stale dataset forever once new source data (or a build_sft_dataset.py
    fix) arrives after a build already ran once: the real failure mode this
    guards is a fully "done" SFT checkpoint, trained against the old
    verbatim-copy only data, being treated as done forever even after the
    real Simple Wikipedia paraphrase corpus is downloaded, since the old
    .npy files and old checkpoint would otherwise still look complete.
    """
    data_files = [
        os.path.join(OUT_DIR, "sft_train_x.npy"),
        os.path.join(OUT_DIR, "sft_train_y.npy"),
        os.path.join(OUT_DIR, "sft_val_x.npy"),
        os.path.join(OUT_DIR, "sft_val_y.npy"),
    ]
    if not all(os.path.exists(p) for p in data_files):
        return True
    if not os.path.exists(MANIFEST_PATH):
        return True
    if not os.path.exists(TRAINED_RAG_TITLES_PATH):
        return True
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        try:
            stored = json.load(f)
        except json.JSONDecodeError:
            return True
    return stored != _input_fingerprint()


BLOCK_SIZE = 1024
PAD_ID = 50256  # gpt2 <|endoftext|>, unused mid sequence, safe as pad

# tens of thousands per shape, not millions: a "short" SFT stage per the
# schedule, not a second pretraining run. Real repetition count for
# template locking on a 124M model has no sourced magic number (LIMA's
# ~1000 examples is for a 65B model already carrying world knowledge, does
# not transfer down), so this is a reasoned starting point, not a citation,
# flagged as such. Roughly matched across shapes so the model does not
# learn to favor one shape over the others.
#
# N_RAG doubled from the original 30,000 once the RAG shape moved to
# training exclusively on real paraphrases (rag_examples() no longer has a
# copy verbatim fallback): with only 238,363 real Simple Wikipedia matches
# to draw from and a fixed 6,000 iteration training budget, more distinct
# titles trained means fewer repeats of each one on average (train_sft.py's
# get_batch() samples uniformly across the whole shuffled dataset, so a
# bigger dataset at the same iteration count means less exposure per row).
# 60,000 was chosen as a real, deliberate middle point, not a maximum:
# using all 238,363 available matches would drop average exposure per row
# to under 1, most rows would likely never be sampled at all across the
# whole run; 60,000 keeps exposure close to the original level while
# giving the model 60x more distinct trained paraphrases than the 1,003 the
# old mixed copy/paraphrase sampling produced. N_TOOL_REPEATS and
# N_NOMATCH_REPEATS scaled with it, to keep the three shapes roughly
# matched in total row count the way this comment already says they should
# be, not left at their old counts while only the RAG shape grew.
N_RAG = 60_000
N_TOOL_REPEATS = 3750  # 16 real traces * 3750 ~= 60,000
N_NOMATCH_REPEATS = 1500  # (20 greeting + 20 factual refusal) * 1500 ~= 60,000

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


def load_simple_wikipedia_paraphrases():
    """Returns {title: one sentence paraphrase}. Empty dict, not an error, if
    toolstore/download_simple_wikipedia.py has not been run yet: rag_examples()
    only ever samples a title present in this dict (no copy verbatim fallback
    exists there anymore, the RAG shape trains on real paraphrases exclusively
    now), so an empty dict here just means zero RAG shape rows get built, a
    real, working, if degraded, dataset rather than a crash.

    Real, independently human written text, not a paraphrase generated by
    prompting this project's own model: an empirical test found the base
    model hallucinates and drifts off topic within a sentence when prompted
    to paraphrase (in context learning is unreliable well below 1 billion
    parameters, this project's own SPEC.md Part 3 already found the same
    thing for RAG context use, which is why the SFT stage exists at all).
    Simple English Wikipedia is a real, separate Wikimedia project, written
    by human editors, covering many of the same topics in different words,
    the same corpus used in real published text simplification research
    (Coster and Kauchak 2011; WikiAuto, Jiang et al. 2020).

    Measured against the real, full downloaded corpus (241,787 Simple
    Wikipedia titles against the first 2,000,000 rows of the 6,458,670 row
    English corpus): 7.78% of English titles have an exact Simple Wikipedia
    title match, projecting to about 502,000 across the full corpus.

    Cut to its first sentence with the same real first_sentence() chat.py
    uses at inference time, not the whole Simple Wikipedia paragraph: a
    concise trained answer needs to be what the model actually learned to
    produce, not a side effect of cutting a longer one down after the fact.
    """
    paraphrases = {}
    if not os.path.exists(SIMPLE_WIKI_TSV):
        return paraphrases
    with open(SIMPLE_WIKI_TSV, "r", encoding="utf-8") as f:
        for line in f:
            title, text = line.rstrip("\n").split("\t", 1)
            if _is_disambiguation(title, text):
                continue
            paraphrases[title] = first_sentence(text)
    return paraphrases


def rag_examples(limit, paraphrases):
    """Known, accepted limitation, measured by bad-cop against the real
    30,000 sampled rows with the real gpt2 encoder: 9 of them (0.03%, e.g.
    "ISO 639:p", whose prefix alone is 10,430 tokens) are longer than
    BLOCK_SIZE before the trailing "\\n" is reached, so tokenize_example()
    truncates the stop token away and the longest of those end up fully
    masked, contributing no training signal at all. Not fixed here: forcing
    a stop token onto a row cut off mid context would teach the model to
    stop mid summary, and dropping rows would need the dataset rebuilt.
    Same "good enough to ship, not perfectly clean" tolerance tool_examples()
    documents for corpus coincidence collisions. Safe at this scale because
    the loss is computed over the flattened batch (model.py's cross_entropy
    with ignore_index=-1), so a fully masked row costs a slot, not a NaN,
    unless an entire batch were drawn from those 9 rows out of ~90,000.

    Every row here is a real, independently written Simple Wikipedia
    paraphrase, never the summary copied verbatim: only titles present in
    paraphrases are sampled at all, no copy verbatim fallback exists in
    this function anymore. Tried the mixed version first, paraphrase when
    available, copy otherwise, and measured its real coverage: of 30,000
    uniformly sampled rows only 1,003 (3.34%) got a real paraphrase, so the
    model's dominant learned behavior stayed "copy," and a real retrieval
    hit had only a 0.0155% chance of landing on one of those 1,003 specific
    trained titles, since paraphrasing at this model size behaves like per
    article memorization, not a rule that transfers to a title it never
    saw. Training exclusively on paraphrases, instead of diluting that
    small signal with tens of thousands of copy examples, gives the model
    one consistent pattern instead of two competing ones, matching the real
    published precedent for this (Nisioi et al. 2017; Zhang and Lapata
    2017: small LSTM models, smaller than this one, trained via ordinary
    supervised learning, not prompting, on real Wikipedia and Simple
    Wikipedia parallel pairs).

    toolstore/chat.py's build_prompt() only ever builds this shape's prompt
    for a retrieved title actually present in the used_titles this function
    returns, not merely present in the paraphrases dict passed in: this
    function stops at `limit` matches in WIKI_TSV file order, so the real
    trained set is a small, order-dependent subset of every title
    load_simple_wikipedia_paraphrases() could theoretically supply a
    paraphrase for (bad-cop measured a real 178,363-title, 74.8% mismatch
    between the two, worth stating plainly since it is exactly the kind of
    silent gap this project's own README says it does not hide). Any
    retrieval hit for a title outside used_titles is treated as no match
    rather than asking the model to answer about content it was never
    actually shown during training.

    Returns (rows, used_titles): used_titles is written by build() to
    TRAINED_RAG_TITLES_PATH, the file toolstore/chat.py's
    _get_paraphrase_titles() reads, so the inference time gate and the real
    training set can never drift apart the way they did before this file
    existed.
    """
    rows = []
    used_titles = set()
    with open(WIKI_TSV, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= limit:
                break
            title, summary = line.rstrip("\n").split("\t", 1)
            if title not in paraphrases:
                continue
            question = f"What is {title}?"
            prefix = f"Context: {summary}\nQuestion: {question}\nAnswer:"
            full = prefix + " " + paraphrases[title] + "\n"
            rows.append((full, prefix))
            used_titles.add(title)
    return rows, used_titles


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
            full = prefix + " " + call + "\n"
            rows.append((full, prefix))
    return rows


def nomatch_examples(repeats):
    rows = []
    all_examples = NOMATCH_EXAMPLES + FACTUAL_NOMATCH_EXAMPLES
    for _ in range(repeats):
        for question, answer in all_examples:
            prefix = f"Context: (none)\nQuestion: {question}\nAnswer:"
            full = prefix + " " + answer + "\n"
            rows.append((full, prefix))
    return rows


def build():
    paraphrases = load_simple_wikipedia_paraphrases()
    if paraphrases:
        print(f"loaded {len(paraphrases)} Simple Wikipedia paraphrase candidates")
    else:
        print("no Simple Wikipedia corpus found, RAG answers will copy the summary verbatim "
              "(run toolstore/download_simple_wikipedia.py first for real paraphrase training)")

    print("generating RAG shape examples from Wikipedia summaries...")
    rows, used_titles = rag_examples(N_RAG, paraphrases=paraphrases)
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
    # toolstore/chat.py's inference time gate's single source of truth for
    # which titles were really trained, not an approximation it re-derives
    # itself. Written before the manifest, same "must never exist before
    # the files it describes" ordering the manifest comment below documents.
    with open(TRAINED_RAG_TITLES_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(used_titles), f)
    # Written last, after every .npy file above already landed: this is the
    # record dataset_needs_rebuild() trusts to mean "these files really do
    # match these inputs," so it must never exist while the .npy files it
    # describes do not (or do not yet reflect this run).
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(_input_fingerprint(), f)
    print(f"wrote {len(train_x)} train and {len(val_x)} val examples to {OUT_DIR}")


if __name__ == "__main__":
    build()
