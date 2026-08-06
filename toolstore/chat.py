"""SPEC.md Part 4's real chat loop: RAG grounded, tool using, SFT template
aware. This is the thing talk.py explicitly said it was NOT. Model loading,
encode/decode, the checkpoint freshness behavior, and the UTF-8 stdout fix
are cloned verbatim from model/talk.py (already correct and tested there),
only the outer turn logic changes: retrieve -> build the matching SFT
template prefix -> generate -> parse for a CALL: -> either dispatch a tool
or print the answer.

Needs the SFT checkpoint (model/sft/checkpoints/ckpt.pt), not the base one,
same reasoning talk.py documents for the base checkpoint: SFT is what
actually taught the model these three fixed template shapes (RAG,
Context: (none), and Question:/Answer: for tool calls), a base only
checkpoint has never seen them and won't follow this loop's prompts
correctly (see LIMA's "superficial alignment hypothesis", cited in
SPEC.md: SFT teaches format and style, not new knowledge).

load()/answer_question() split from the CLI loop, same reason and same
shape as model/talk.py's own split: gui.py imports this module and calls
into it directly from a worker thread, no subprocess, no stdin piping.
"""
import datetime
import json
import logging
import os
import re
import sys
from contextlib import nullcontext

import torch
import tiktoken
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from model import GPTConfig, GPT

from query import retrieve, get_reranker
from tools import TOOLS
from web_search import web_search
from sentence_boundary import first_sentence, SENTENCE_BOUNDARY_RE, MIN_SENTENCE_WORDS

# Windows' console default codepage (cp1252) mangles the curly quotes and
# dashes tiktoken's gpt2 vocab decodes into, same fix as talk.py.
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(__file__)

# A dedicated named logger with its own FileHandler, not gui.py's
# logging.basicConfig(): basicConfig configures the root logger once,
# process wide, and gui.py may already have called it first when both
# modules load in the same process, so attaching a handler straight to this
# module's own logger is what actually gets a file out of it, and doing it
# this way means chat.py's logging works the same whether gui.py loaded it
# or not (the CLI loop at the bottom of this file, or a bare test import).
_LOG_DIR = os.path.join(HERE, "..", "logs")


class ChatTurnFileHandler(logging.FileHandler):
    """A FileHandler that reports a failed write instead of letting it
    disappear into logging's own error path.

    logging.Handler.emit() (inherited from StreamHandler) wraps its own
    stream.write() in try/except and hands any failure to handleError(),
    whose default implementation dumps a raw "--- Logging error ---"
    traceback to stderr and returns normally. So a real OSError (a full
    disk, a revoked permission on chat_turns.log) produces no ERROR line
    naming which subsystem lost what: measured, a simulated disk full write
    produced 1421 bytes of uncaptured stderr and zero logger.error() calls.

    Overriding handleError is the documented extension point for exactly
    this ("You could, however, replace this with a custom handler if you
    wish", logging.Handler.handleError's own docstring), and the stdlib
    overrides it itself in logging.handlers.SocketHandler for the same kind
    of reason.

    What this override must NOT do is raise, which is how an earlier version
    pushed the failure out to _log_turn's except block. Logger.callHandlers()
    is one loop over "this logger's handlers, then every ancestor logger's
    handlers" calling hdlr.handle(record) with no try/except of its own
    (CPython Lib/logging/__init__.py, `while c: for hdlr in c.handlers`), and
    Handler.handle() only wraps emit() in try/finally to release its lock, so
    anything raised out of handleError aborts that walk where it stands:
    every handler after the failing one, including the root logger's handler
    gui.py installs with logging.basicConfig() in this same process, never
    sees the record. Measured against the raising version, in exactly that
    topology: the root handler's buffer was empty for a turn whose file write
    failed. Reporting in place and returning normally is the contract every
    stdlib handler keeps (SocketHandler.handleError closes its socket and
    returns, Handler.handleError prints and returns), and it is what keeps a
    failure of THIS handler local to this handler.

    _failure_being_reported is the recursion guard that makes reporting
    through this same logger safe: the ERROR line is routed straight back
    into this handler, and on a genuinely broken file that write fails too,
    so the second failure falls back to stderr instead of reporting again.
    It is only touched from handleError, which runs inside emit(), which
    Handler.handle() calls while holding this handler's own RLock, so it is
    guarded by the same lock the stream is, and re-entering on the same
    thread re-acquires that RLock rather than deadlocking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._failure_being_reported = None

    def handleError(self, record):
        exc = sys.exc_info()[1]
        if self._failure_being_reported is not None:
            _print_log_failure_to_stderr(self._failure_being_reported, exc)
            return
        self._failure_being_reported = exc
        try:
            _report_log_failure(exc)
        finally:
            self._failure_being_reported = None


logger = logging.getLogger("chat")
logger.setLevel(logging.INFO)
if not logger.handlers:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        _handler = ChatTurnFileHandler(os.path.join(_LOG_DIR, "chat_turns.log"),
                                       encoding="utf-8")
        _handler.setFormatter(logging.Formatter("%(message)s"))
    except OSError as _log_setup_error:
        # Turn logging is a side effect of chatting, never a precondition for
        # it, and this runs at import: a logs/ path that is not a writable
        # directory (occupied by a file, permissions revoked) would otherwise
        # raise straight out of `import chat` and take gui.py's Chat tab and
        # the CLI loop down with it. Degrade to stderr instead, at WARNING so
        # the failure itself is visible without every turn's full block (RAG
        # context included) spamming the console it is now falling back to.
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setLevel(logging.WARNING)
        logger.addHandler(_handler)
        logger.error("chat turn logging to %s is disabled: %s",
                     _LOG_DIR, _log_setup_error, exc_info=True)
    else:
        logger.addHandler(_handler)

# -----------------------------------------------------------------------------
out_dir = os.path.join(HERE, "..", "model", "sft", "checkpoints")
block_size = 1024
max_new_tokens = 100  # answers are short (a sentence, or one CALL: line), not
                       # a 200 token completion like talk.py's raw sampling
temperature = 0.8
top_k = 200
# Nucleus sampling (Holtzman et al. 2019, arXiv:1904.09751), added to
# model.py's generate() as a new, backward compatible top_p=None default;
# 0.9 is the value both that paper and HuggingFace's own generation
# defaults converge on. Composes with top_k above (top_k filters first,
# top_p narrows further), free per token, applied to every generate() call
# in this file, not just the RAG shapes N_BEST_OF below is scoped to.
top_p = 0.9
# Best-of-N reranking, RAG shapes only (real Context: content, vector doc
# or live search result): generates N_BEST_OF real candidates and keeps
# the one query.py's already-loaded CrossEncoder scores as most relevant
# to the question, the identical (query_text, content) -> predict() shape
# query.py's own search() already uses for RAG chunk reranking, just
# re-pointed at generated candidates instead of retrieved chunks. Not
# applied to tool-call or refusal generations: dispatch_call()'s own
# docstring already makes the reason explicit for tool calls ("a second
# generate() call could only add a chance to say something wrong on top of
# a right answer"), and a CrossEncoder trained for question/passage
# relevance has no meaningful signal to rank refusal candidates against
# each other.
N_BEST_OF = 4
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# -----------------------------------------------------------------------------

_model = None
_encode = None
_decode = None
_ctx = None

TRAINED_RAG_TITLES_PATH = os.path.join(HERE, "..", "model", "sft", "trained_rag_titles.json")
_paraphrase_titles = None


def _get_paraphrase_titles():
    """The exact set of titles model/sft/build_sft_dataset.py's
    rag_examples() actually turned into an SFT training row, loaded once
    and cached: a title, not the full text, is all build_prompt() needs to
    decide whether a retrieval hit is safe to answer from.

    Reads build()'s own TRAINED_RAG_TITLES_PATH output rather than
    re-deriving an approximation from the raw Simple Wikipedia TSV (the
    first version of this function did that: every title clearing the
    disambiguation filter, not just the ones rag_examples() actually
    sampled). bad-cop measured the real gap that approach left open:
    rag_examples() streams WIKI_TSV in file order and stops at N_RAG
    matches, so 178,363 of 238,363 candidate titles (74.8%) cleared the old
    gate while never having been shown to the model during training —
    exactly the "per article memorization, not a rule that transfers to a
    title it never saw" failure the whole no-copy-fallback feature exists
    to close, reopened for most of the gate's own accepted titles. Reading
    build()'s real output instead of re-deriving a guess makes the two
    impossible to drift apart: there is only one computation of "which
    titles were trained," not two that have to be kept in sync by hand.

    Empty set, not an error, if the file does not exist yet (before the
    first real build, or an old dataset built before this file existed):
    every retrieval hit then falls through to the no match branch, same as
    it would if the vector store itself were empty, a real, working,
    maximally conservative state rather than a crash.
    """
    global _paraphrase_titles
    if _paraphrase_titles is None:
        titles = set()
        if os.path.exists(TRAINED_RAG_TITLES_PATH):
            with open(TRAINED_RAG_TITLES_PATH, "r", encoding="utf-8") as f:
                titles = set(json.load(f))
        _paraphrase_titles = titles
    return _paraphrase_titles


# Same fixed, low entropy template the model was actually trained on
# (model/sft/build_sft_dataset.py), a strict regex is enough, no real parser
# needed (SPEC.md Part 4's own reasoning, confirmed against LangChain's real
# production MRKL output parser using the same re.search + fallback shape
# against far less constrained text).
CALL_RE = re.compile(r'CALL:\s*(\w+)\((.*)\)')

# The exact, fixed refusal text model/sft/build_sft_dataset.py's
# FACTUAL_NOMATCH_EXAMPLES trains for every real factual question the
# vector store has no confident match for (greetings/small talk train a
# different, varied reply for the same Context: (none) shape, and are left
# alone here, a DuckDuckGo search for "hi" would just return noise). Startswith,
# not equality: generation can still trail off past this fixed sentence.
NOMATCH_REFUSAL_PREFIX = "I don't have information about that"

# What the user is shown when DuckDuckGo genuinely returned nothing.
# web_search() reports that case as None, a value no page body can forge,
# and this file owns the wording, next to the "[web search error: ...]"
# text run_web_search() builds for the other diagnostic outcome.
NO_RESULTS_MESSAGE = "[web search: no results found]"


def is_loaded():
    """True once load() has actually loaded a checkpoint. See talk.py's
    own is_loaded() for why a caller forcing device must check this first,
    same reasoning, same bug, same fix.
    """
    return _model is not None


def load():
    """Loads the SFT checkpoint. Safe to call more than once, only does
    real work the first time. Raises FileNotFoundError if no SFT checkpoint
    exists yet, callers decide how to show that.
    """
    global _model, _encode, _decode, _ctx
    global out_dir, block_size, max_new_tokens, temperature, top_k, seed, device, dtype
    if _model is not None:
        return

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        exec(open(os.path.join(HERE, "..", "model", "configurator.py")).read())
    finally:
        sys.argv = saved_argv

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    _ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No SFT checkpoint found at {ckpt_path} yet, nothing to chat with. "
            "Base pretraining and SFT fine tuning both need to finish first."
        )

    print(f"Loading {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"(fine tuned to iteration {checkpoint['iter_num']})")
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    _model = model

    enc = tiktoken.get_encoding("gpt2")
    _encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    # model_args.vocab_size is 50304 (nanoGPT pads up to a multiple of 64 for GPU
    # efficiency), tiktoken's real gpt2 vocab is only 50257 (ids 0-50256), so the
    # model can technically sample 47 ids tiktoken can't decode (confirmed by
    # good-cop: a real KeyError crash, reproduced against a random-init model,
    # real training pushes these ids' probability toward zero since they're
    # never a real training target, but that's not a hard guarantee). Same shape
    # talk.py/sample.py inherit unmodified from nanoGPT; filtering at decode
    # here rather than touching those shared reference files.
    _decode = lambda l: enc.decode([t for t in l if t < enc.n_vocab])


def truncate_to_tokens(text, max_tokens):
    """encode -> slice -> decode, the accepted pattern (OpenAI's own
    cookbook uses this shape too, tiktoken has no built in truncate helper,
    a real open feature request, not yet merged), not a hand rolled
    character count estimate which would not track real token count.
    """
    ids = _encode(text)
    if len(ids) <= max_tokens:
        return text
    return _decode(ids[:max_tokens])


def _truncate_question(question, context_text):
    """Cuts the question down to whatever the template budget leaves once
    `context_text`, the fixed template words, and max_new_tokens of answer
    are all paid for. The question is the last thing to be cut, and it is
    cut here rather than left to generate().

    Without this, a question long enough to blow the budget on its own is
    still truncated, just implicitly and from the wrong end:
    model.py's generate() crops to `idx[:, -block_size:]`, from the FRONT,
    so what silently disappears is the "Context: ...\\nQuestion: " prefix,
    the one part of the prompt the SFT template is built out of. The model
    then sees an unrecognizable tail of raw text with no template at all
    and free-associates a fluent, confident, wrong sentence instead of the
    trained refusal. Measured against the real checkpoint: a 5000 character
    question encodes to 1262 tokens against block_size 1024, and produced
    "The words are a list of words, which are written by a computer
    algorithm." rather than NOMATCH_REFUSAL_PREFIX.

    Truncating the input before it reaches the model, instead of letting
    the model's own window silently do it, is the same contract
    HuggingFace tokenizers give with `truncation=True, max_length=...`
    (right side by default, so the front of the prompt survives), and the
    same encode -> slice -> decode shape truncate_to_tokens() already uses
    for the Context: block.
    """
    budget = block_size - max_new_tokens - len(_encode(f"Context: {context_text}\nQuestion: \nAnswer:"))
    return truncate_to_tokens(question, max(budget, 0))


def _context_prompt(question, content):
    """Builds a Context:-having prompt for any retrieved text, a vector
    store passage or a live search result alike, truncated to the same
    real token budget build_prompt() already enforces for retrieve()'s
    docs (SPEC.md Part 4 step 6). Returns (prefix, context_text);
    context_text is what was actually fed in, not the raw, untruncated
    input, same "show what the model really saw" reasoning build_prompt()
    already documents for its own context_text return value.

    The retrieved content is what gets cut first, all the way to nothing if
    that is what the question needs. Only a question that still does not
    fit against an empty Context: block is itself cut, so every prompt that
    fits today is built exactly as it was.
    """
    budget = block_size - max_new_tokens - len(_encode(f"Context: \nQuestion: {question}\nAnswer:"))
    if budget < 0:
        # There is no passage left to give back: the question alone
        # overflows the template, so it is the question that has to be cut
        # (see _truncate_question for what generate() does otherwise).
        question = _truncate_question(question, "")
        budget = 0
    context_text = truncate_to_tokens(content, budget)
    prefix = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
    return prefix, context_text


def _no_context_prompt(question):
    """The Context: (none) half of the same template, for a real retrieval
    miss, a paraphrase-gated hit, or a mode that never retrieved at all.
    Truncated the same explicit way _context_prompt() truncates its own
    inputs: this shape has no passage to cut, so the question is measured
    directly against the "(none)" template's own overhead.
    """
    return f"Context: (none)\nQuestion: {_truncate_question(question, '(none)')}\nAnswer:"


def _generate(prefix):
    """Runs one real generate() pass and returns the decoded, trimmed
    answer. load() must already have been called. Factored out of
    _run_turn() so the live search fallback can ask the model to generate
    a real, on the spot answer from a search result the same way it
    already does for a retrieved vector store passage, rather than ever
    printing external text untouched: "never copy and paste the RAG
    result" applies the same way to a live search hit once one exists,
    not only to what retrieve() finds.
    """
    with torch.no_grad():
        with _ctx:
            start_ids = _encode(prefix)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            y = _model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    generated_ids = y[0].tolist()[len(start_ids):]
    return _decode(generated_ids).split("\n")[0].strip()


def _best_of_n_generate(prefix, question, n=N_BEST_OF):
    """Generates n real candidates via _generate() and keeps the one
    query.py's already-loaded CrossEncoder scores as most relevant to the
    question, the identical (query_text, content) -> reranker.predict()
    shape query.py's own search() already uses to rerank retrieved RAG
    chunks (cloned from there, not reinvented), just re-pointed at
    generated candidates instead of retrieved chunks. Only ever called
    from a Context:-having prompt (a real vector doc or live search
    result), where "how relevant is this candidate to the question" is a
    meaningful signal; never from a tool-call or refusal generation, see
    the N_BEST_OF module comment for why.

    n=1 skips the reranker call entirely, not just short circuits after
    one candidate: real code paths with N_BEST_OF overridden to 1 (a test,
    or a future config change) should not pay for a CrossEncoder load that
    was never going to change the outcome.
    """
    candidates = [_generate(prefix) for _ in range(n)]
    if n <= 1:
        return candidates[0]
    reranker = get_reranker()
    pairs = [(question, c) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    best_i = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_i]


def build_prompt(question, use_retrieval=True):
    """Picks whichever SFT template shape applies and returns (prefix,
    citation, context_text) where citation and context_text are both None
    for the no match shape. Only the single best reranked doc is used even
    though retrieve() can return up to five: the SFT template has room for
    exactly one Context: block, a real integration constraint found by
    reading build_sft_dataset.py's rag_examples(), not a simplification of
    convenience.

    context_text is the actual, possibly truncated text substituted into
    the Context: block, not citation["content"] (the full, untruncated
    retrieved passage): the GUI's RAG context viewer shows what the model
    really saw, not the whole source article.

    use_retrieval=False skips retrieve() and goes straight to the same
    Context: (none) shape retrieve() finding nothing would produce, for
    mode="web": the vector store is what that mode turns off, not the model,
    so the model still gets a real turn to dispatch a tool or give its own
    trained reply, and only a genuine refusal falls through to a live
    search, same as every other mode already does.
    """
    docs = retrieve(question) if use_retrieval else []
    if not docs:
        return _no_context_prompt(question), None, None

    best = docs[0]
    if best["metadata"]["title"] not in _get_paraphrase_titles():
        # build_sft_dataset.py's rag_examples() only ever trains the RAG
        # shape on titles with a real Simple Wikipedia paraphrase (no copy
        # fallback, per the "never copy and paste the RAG result" rule); a
        # retrieval hit outside that trained set gets the exact same
        # Context: (none) treatment a true miss would, so the model is never
        # asked to paraphrase content it was never taught to.
        return _no_context_prompt(question), None, None
    prefix, context_text = _context_prompt(question, best["content"])
    return prefix, best, context_text


def dispatch_call(match):
    """Looks up the tool, runs it, catches any exception (SPEC.md Part 4's
    own explicit instruction: a from scratch small model's output cannot be
    trusted, a malformed or unsupported call must not crash the loop).
    Returns the line to print, real tool output, not a second model pass:
    the tool already gives the real, correct value, a second generate() call
    could only add a chance to say something wrong on top of a right answer.
    """
    name, args = match.group(1), match.group(2)
    tool = TOOLS.get(name)
    if tool is None:
        return f"[tool: {name} — unknown tool, ignoring]"
    try:
        result = tool(args) if args.strip() else tool()
    except Exception as e:
        return f"[tool: {name}({args}) — error: {e}]"
    return f"[tool: {name}] Result: {result}"


def run_web_search(query):
    """Runs the live search, catching any exception so a failed lookup ends
    the turn, not the session. Same guard and same reason as dispatch_call()
    right above, at the other untrusted external I/O boundary in this file:
    web_search() normalizes every ddgs failure to WebSearchError, and
    catching Exception rather than that one type is what still holds if a
    failure ever escapes that normalization (web_search.py's docstring
    records a real, unmocked failure where an exception that was not a
    DDGSException killed the whole chat session, and gui.py calls
    answer_question() from a worker thread where a raise loses the tab's
    answer too).

    Returns (text, is_diagnostic): is_diagnostic is True for a failed
    lookup and for a search that genuinely found nothing, neither one real
    search content there is anything to generate an answer from, False for
    a real result. The flag comes from which of web_search()'s three
    outcomes happened (a raise, None, or a result string), never from
    inspecting the result text: a page body is unconstrained,
    attacker/DDG-controlled text, so a real hit whose body opened with
    "[web search error:" used to be classified as a diagnostic and shown to
    the user untouched by the model, which bad-cop reproduced (one
    generate() call for a turn that owed two). Both user facing diagnostic
    strings are built here, the layer that talks to the user, rather than
    half here and half in web_search.py.
    """
    try:
        text = web_search(query)
    except Exception as e:
        # Reported, not just returned: the returned string tells the user
        # this turn failed, and the traceback tells whoever reads the log
        # which external call failed and why, the same way
        # _report_log_failure() does for a lost turn log.
        logger.error("live search failed: %s", e, exc_info=True)
        return f"[web search error: {e}]", True
    if text is None:
        return NO_RESULTS_MESSAGE, True
    return text, False


_LOG_SEPARATOR = "=" * 80


def _log_turn(question, mode, result):
    """Appends one human readable block to logs/chat_turns.log: the
    question, the mode, the full RAG context or live search result text
    when this turn actually produced one (not just the short final answer
    that already gets cut to a sentence or two), and the real tagged
    answer. Every turn gets a block, including a tool call or a plain
    "hi", not only the ones with a source, so the log is a complete record
    of what was asked and answered, not a filtered one someone has to
    guess the gaps in; the RAG/live search sections are the only part
    that's conditional, since a trivial turn has nothing there to show.

    Logging is a side effect of already having the answer, never the
    reason a turn could fail: any error is reported through this same
    logger at ERROR level instead of raising, so a full disk or a
    permissions problem loses a log line, never the answer the user is
    waiting on. The except block below covers failures on this side of the
    logger, building the block itself (a result dict missing a key, an
    unformattable value). A failure of the file write is reported by
    ChatTurnFileHandler.handleError instead, because the record still has
    other handlers to reach after this one fails; see that class, and note
    that a plain logging.FileHandler would report neither.
    """
    try:
        lines = [
            _LOG_SEPARATOR,
            f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | mode={mode} | Q: {question}",
            "-" * 80,
        ]
        if result.get("context"):
            title = result.get("source_title") or "(untitled)"
            lines.append(f"RAG CONTEXT (source: {title}):")
            lines.append(result["context"])
            lines.append("")
        if result.get("refused_context"):
            # The vector passage the model was shown and refused before
            # falling back to a live search: "context" above is what
            # actually produced the visible answer once that fallback
            # fires (the live search text), so without this section the
            # refused passage would be gone from the record the moment the
            # fallback happens, contradicting this function's own "complete
            # record of what was asked and answered" promise above.
            lines.append(f"REFUSED LOCAL CONTEXT (source: {result.get('refused_source_title') or '(untitled)'}):")
            lines.append(result["refused_context"])
            lines.append("")
        if result.get("web_result"):
            lines.append("LIVE SEARCH RESULT (DuckDuckGo):")
            lines.append(result["web_result"])
            lines.append("")
        lines.append("ANSWER:")
        lines.append(result["text"])
        logger.info("\n".join(lines) + "\n")
    except Exception as e:
        _report_log_failure(e)


def _report_log_failure(exc):
    """Reports a failed turn log write at ERROR level through this module's
    own logger, falling back to stderr when reporting it raises.

    The single place a lost turn log is reported from, called by the only
    two things that can see one: ChatTurnFileHandler.handleError, when the
    write itself failed, and _log_turn's except block, when building the
    block failed before any write was attempted.

    The fallback is not defensive padding, though it is no longer the
    ordinary path: ChatTurnFileHandler now contains its own write failures,
    but this logger propagates to the root logger (gui.py's
    logging.basicConfig() handler, in the same process), and any handler
    reached that way is free to raise straight out of
    Logger.callHandlers(). This call runs inside _log_turn's except block,
    whose whole job is keeping a logging failure away from the answer the
    user is waiting on. One level only: the fallback writes to stderr
    directly and never re-enters the logger.
    """
    try:
        logger.error("failed to log chat turn: %s", exc, exc_info=True)
    except Exception as report_exc:
        _print_log_failure_to_stderr(exc, report_exc)


def _print_log_failure_to_stderr(exc, report_exc):
    """The last resort both guards share: chat.py's own named line, rather
    than stdlib logging's anonymous "--- Logging error ---" traceback, on
    the console a maintainer actually reads during a disk full incident.
    """
    print(f"chat: failed to log chat turn ({exc!r}), and reporting that "
          f"through the log failed too ({report_exc!r})", file=sys.stderr)


def answer_turn(question, mode="auto"):
    """One full turn: routes web: prefixed input straight to DuckDuckGo,
    otherwise retrieves, builds the matching prompt, generates, and ends in
    exactly one of four outcomes, in this order: a dispatched tool call, a
    live search when the model gave its trained refusal, the retrieved
    source's answer, or the model's own unsourced answer. calls load() first
    if the checkpoint hasn't been loaded yet.

    mode picks which of those four outcomes are reachable, for the GUI's
    Chat tab toggle:
      "auto"   default, the behavior above, unchanged.
      "web"    turns off the vector store, not the model: the model still
               runs and can still dispatch a real tool call or give its own
               trained reply, it just never sees a Context: block from
               retrieve(), so the only way this mode reaches a live search
               is a genuine refusal, same as "auto" does. Tried first as
               "skip the model entirely, always search," found by testing
               that this was worse: a bare live search for a tool question
               like "what is 17 times 23" returns a random calculator
               website instead of the real computed answer, and a bare live
               search for a greeting like "hi" returns whatever DuckDuckGo
               ranks first for that word, unrelated noise.
      "vector" turns off the live search, not the vector store (gui.py
               labels it "Vector only"): retrieval still runs and the model
               still sees a real Context: block, tool calls and sourced
               answers behave the same, but a refusal is returned as is
               instead of falling back to a live search, for someone who
               wants to see what the from scratch model does on the local
               corpus alone. A refusal there is tagged with which of the two
               things happened, retrieval finding nothing or the model
               refusing what it found, the same distinction "auto" draws.
      "model"  closed-book (the standard term for a QA system answering
               from model weights only, no retrieval — see e.g. arXiv
               2607.21861): both the vector store AND the live search
               fallback are off, tool calls still dispatch. Neither "web"
               nor "vector" alone was this — "web" turns retrieval off but
               still falls back to a live search on refusal, "vector" never
               searches but still runs retrieval. This is their
               intersection, added on direct instruction ("I'd rather the
               AI be bad than not use the AI, I need to show I made it"):
               an answer under this mode is never anything but the model's
               own trained weights, whatever the question. gui.py's Source
               toggle labels it "Model only, never search".
    The web: prefix is checked before mode either way, so it stays a real
    override that works no matter what the toggle is set to: it is an
    explicit, typed command to search, not an automatic fallback, so it is
    not what "model" mode's "never search" promise is about.

    Returns a dict, not just the printable string, so the GUI's RAG context
    viewer can show the exact text that was fed into the model without
    re-deriving it, and so every turn can be logged in full: {"text": the
    same printable response answer_question() used to return directly,
    "context": the exact Context: block text fed into the prompt that
    actually produced the visible answer this turn, or None on any branch
    that never used one, "source_title": that context's title ("live
    search result" whenever a live search fed the model, whether from the
    web: prefix or the refusal fallback), or None, "web_result": the raw
    live search result whenever run_web_search() was actually called this
    turn, or None everywhere else, including mode="vector"'s refusal, which
    never searches, "refused_context"/"refused_source_title": only set on
    the "auto"/"web" refusal-then-search fallback when a real vector source
    was refused first, the passage and title that got refused before the
    live search took over as "context" above, so it stays a real,
    recoverable part of the turn instead of vanishing the moment the
    fallback fires; None everywhere else}.

    Every turn is logged to logs/chat_turns.log as a full, human readable
    block (question, mode, the real context/web result text if there was
    one, the final tagged answer), not just this short printable string:
    the whole point of the GUI's RAG viewer is seeing exactly what the
    model or a live search really returned, and that should survive after
    the app closes, not just live in memory for one session.
    """
    result = _run_turn(question, mode)
    _log_turn(question, mode, result)
    return result


def _live_search_answer(question):
    """Runs a live search and, if it produced real content, asks the model
    to generate a real answer from it on the spot (the "retrieve, then
    generate" contract, not retrieve-then-print, per the explicit "talk to
    ai using rag should ALWAYS get a response made with AI generated on the
    spot" instruction). run_web_search()'s is_diagnostic flag (a failed
    lookup, or no results at all) means there is no real search content,
    only diagnostic text a user needs to read verbatim, not a passage to
    hand the model and ask it to paraphrase, so that case is returned as
    is, nothing generated.

    Returns (answer, context_text, result): answer is either the model's
    real generated text or the diagnostic string, context_text is the
    truncated text actually fed to the model (None on the diagnostic path,
    nothing was fed), result is exactly what run_web_search() returned
    either way, for the "web_result" field and logging.
    """
    result, is_diagnostic = run_web_search(question)
    if is_diagnostic:
        return result, None, result
    load()
    prefix, context_text = _context_prompt(question, result)
    answer = first_sentence(_best_of_n_generate(prefix, question))
    return answer, context_text, result


def _run_turn(question, mode):
    if question.lower().startswith("web:"):
        query = question[len("web:"):].strip()
        answer, context_text, result = _live_search_answer(query)
        if context_text is None:
            text = answer
        else:
            text = f"{answer}\n[source: live search result (DuckDuckGo), AI generated]"
        return {"text": text, "context": context_text,
                "source_title": "live search result" if context_text is not None else None,
                "web_result": result}

    load()
    prefix, source, context_text = build_prompt(question, use_retrieval=(mode not in ("web", "model")))
    # Best-of-N only for a real Context:-having prompt (source is not None,
    # already gated in build_prompt() to a title the model was actually
    # trained to paraphrase): that's the only shape a question/candidate
    # relevance score means anything for. A "Context: (none)" prompt (tool
    # question, greeting, real no-match) gets the single, cheaper
    # _generate() call it always has, same reasoning N_BEST_OF's own
    # module comment gives for skipping tool calls and refusals.
    answer = _best_of_n_generate(prefix, question) if source is not None else _generate(prefix)

    call_match = CALL_RE.search(answer)
    if call_match:
        return {"text": dispatch_call(call_match), "context": None, "source_title": None, "web_result": None}
    elif answer.startswith(NOMATCH_REFUSAL_PREFIX):
        # The model gave its trained "I don't know" refusal, so this was a
        # real factual question it can't answer, not small talk. Checked
        # BEFORE `source is not None` on purpose, mode or no mode: retrieve()
        # clearing the rerank threshold does not mean the model actually
        # used the context, and pairing a refusal with a [source: ...] tag
        # is self contradictory to the user, whatever the toggle is set to.
        # The model's own refusal is the authoritative signal that no local
        # answer exists, whatever retrieval thought.
        source_title = source["metadata"]["title"] if source is not None else None
        if mode in ("vector", "model"):
            # What "vector" turns off is the live search fallback, not the
            # vector store (gui.py's Source toggle labels it "Vector only",
            # against "Model only, web if no match" for mode="web"), so the
            # refusal is the real answer here rather than a cue to fetch one
            # elsewhere. Which of two different events produced it still has
            # to be named honestly: retrieval coming back empty is not the
            # same as retrieval handing the model a confident passage that it
            # then refused, and saying the first when the second happened
            # contradicts this file's contract that every answer says where
            # it came from. Same distinction and same wording as the fallback
            # tag below, which has drawn it since this branch was added.
            #
            # "model" (gui.py's "Model only, never search") is the real
            # closed-book mode this same stop-here behavior was missing
            # before: retrieval is off too (build_prompt() call above), so
            # source is always None here and "no confident source found"
            # would falsely imply retrieval ran and came up empty. Named
            # honestly instead: nothing was ever looked up, on either side.
            if mode == "model":
                why = "vector store and live search both off"
            else:
                why = "model refused the local match" if source is not None else "no confident source found"
            text = f"{answer}\n[{why} — answered from the model's own training only]"
            # A refused context was still really fed to the model this turn,
            # so the GUI's viewer can show what it saw, exactly as it already
            # does for the sourced branch under this same mode. Withholding it
            # only here would make one mode's context viewer work on one
            # branch and not the other, for no reason the toggle's own label
            # supports.
            return {"text": text, "context": context_text, "source_title": source_title, "web_result": None}
        # Fall back to a live search, but the response shown is still the
        # model generating a real answer on the spot from that search
        # result, the same "retrieve, then generate" contract this file
        # already holds for the vector store, not the search text printed
        # through untouched (the original, explicit design here, changed on
        # direct instruction: "talk to ai using rag should ALWAYS get a
        # response made with AI generated on the spot" — retrieve-then-print
        # was not that, whichever source did the retrieving).
        search_answer, search_context_text, result = _live_search_answer(question)
        # Tag the real reason: retrieval may well have found a source that
        # the model then refused to use, and this file's whole contract is
        # that every answer says where it came from. mode="web" needs its own
        # wording because source is None there for a different reason than it
        # is under "auto": the vector store was never consulted at all, and
        # "no local match" would claim it was searched and came back empty.
        if mode == "web":
            why = "vector store off"
        elif source is not None:
            why = "model refused the local match"
        else:
            why = "no local match"
        if search_context_text is None:
            # The search failed or found nothing, so there was nothing to
            # generate from; search_answer is already run_web_search()'s own
            # diagnostic text, shown verbatim same as the web: prefix path.
            text = f"{search_answer}\n[{why}, fell back to live search]"
        else:
            text = f"{search_answer}\n[{why}, fell back to live search, AI generated from the search result]"
        return {"text": text, "context": search_context_text,
                "source_title": "live search result" if search_context_text is not None else source_title,
                "web_result": result,
                # bad-cop found this branch's own "context" field no longer
                # carries what the model actually saw and refused, once it
                # was repointed at the live search text that produced the
                # shown answer (both cannot be "context" at once, and the
                # visible answer's own real basis wins that slot): these two
                # keep the refused vector passage recoverable, in the turn
                # log _log_turn() writes (gui.py's context viewer renders
                # "context" only) and to any caller reading the dict,
                # matching _log_turn()'s own "complete record" docstring,
                # instead of it becoming unrecoverable the moment the live
                # search fallback fires. None when there was no real vector
                # source to refuse in the first place.
                "refused_context": context_text if source is not None else None,
                "refused_source_title": source_title}
    elif source is not None:
        # rag_examples() trains this shape to continue with the FULL Wikipedia
        # summary as the answer (model/sft/build_sft_dataset.py), which reads
        # as a paragraph, not a concise answer. Cut to the first sentence only
        # here, not on the other three branches: tool-call args can contain
        # "." or "?" that would corrupt CALL_RE's own match if trimmed first,
        # and the no-match/greeting shape ("Hello! What would you like to
        # know?") is already trained short and a real two sentence reply, not
        # excess to cut.
        answer = first_sentence(answer)
        text = f"{answer}\n[source: {source['metadata']['title']}, rerank score: {source['score']:.2f}]"
        return {"text": text, "context": context_text, "source_title": source["metadata"]["title"], "web_result": None}
    else:
        # Same distinction the fallback branch above already draws, and the
        # same reason: under mode="web"/"model" nothing was ever sought, so
        # "no confident source found" would claim a search happened and came
        # up empty when retrieve() was never even called.
        if mode == "web":
            why = "vector store off"
        elif mode == "model":
            why = "vector store and live search both off"
        else:
            why = "no confident source found"
        text = f"{answer}\n[{why} — answered from the model's own training only]"
        return {"text": text, "context": None, "source_title": None, "web_result": None}


def answer_question(question, mode="auto"):
    """Backward compatible string-only wrapper around answer_turn(), the
    original shape the CLI loop and every existing test still use. The GUI's
    Chat tab calls answer_turn() directly instead, for the "context" value.
    """
    return answer_turn(question, mode)["text"]


if __name__ == "__main__":
    try:
        load()
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    print("=" * 70)
    print("Chat with the from-scratch model. Fine tuned on a small, fixed set of")
    print("templates: it can answer from a matching Wikipedia summary, run the")
    print("calculator or current_datetime tool, or say it doesn't have a match.")
    print("It has NOT seen general instructions, multi-turn context, opinions,")
    print("code, or anything outside those templates — free-form questions")
    print("outside this scope will likely get a poor or off-template answer.")
    print("Every response below is tagged with where it came from.")
    print("Type 'web: <question>' to search DuckDuckGo instead of the local")
    print("corpus: the search result is fed to the model and it generates a")
    print("real answer from it on the spot (the model was never trained on")
    print("web search results, so this can be lower quality than a sourced")
    print("local answer).")
    print("Empty line or Ctrl+C to quit.")
    print("=" * 70)
    print()

    while True:
        try:
            question = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question.strip():
            break
        print(answer_question(question))
        print()
