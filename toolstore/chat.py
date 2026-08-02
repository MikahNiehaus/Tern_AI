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
import os
import re
import sys
from contextlib import nullcontext

import torch
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from model import GPTConfig, GPT

from query import retrieve
from tools import TOOLS
from web_search import web_search

# Windows' console default codepage (cp1252) mangles the curly quotes and
# dashes tiktoken's gpt2 vocab decodes into, same fix as talk.py.
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(__file__)

# -----------------------------------------------------------------------------
out_dir = os.path.join(HERE, "..", "model", "sft", "checkpoints")
block_size = 1024
max_new_tokens = 100  # answers are short (a sentence, or one CALL: line), not
                       # a 200 token completion like talk.py's raw sampling
temperature = 0.8
top_k = 200
rerank_threshold = 7.0  # same gate query.py's own retrieve() default uses,
                         # named here too since it drives the banner/citation text
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# -----------------------------------------------------------------------------

_model = None
_encode = None
_decode = None
_ctx = None

# Same fixed, low entropy template the model was actually trained on
# (model/sft/build_sft_dataset.py), a strict regex is enough, no real parser
# needed (SPEC.md Part 4's own reasoning, confirmed against LangChain's real
# production MRKL output parser using the same re.search + fallback shape
# against far less constrained text).
CALL_RE = re.compile(r'CALL:\s*(\w+)\((.*)\)')


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


def build_prompt(question):
    """Picks whichever SFT template shape applies and returns (prefix,
    citation) where citation is None for the no match shape. Only the
    single best reranked doc is used even though retrieve() can return up
    to five: the SFT template has room for exactly one Context: block, a
    real integration constraint found by reading build_sft_dataset.py's
    rag_examples(), not a simplification of convenience.
    """
    docs = retrieve(question)
    if not docs:
        prefix = f"Context: (none)\nQuestion: {question}\nAnswer:"
        return prefix, None

    best = docs[0]
    # room for the rest of the template plus the answer itself, not just the
    # context block, same truncation budget SPEC.md Part 4 step 6 describes
    budget = block_size - max_new_tokens - len(_encode(f"Context: \nQuestion: {question}\nAnswer:"))
    context_text = truncate_to_tokens(best["content"], max(budget, 0))
    prefix = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
    return prefix, best


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


def answer_question(question):
    """One full turn: routes web: prefixed input straight to DuckDuckGo,
    otherwise retrieves, builds the matching prompt, generates, and either
    dispatches a tool call or returns the tagged answer. Returns the full
    printable response (including the trailing [source: ...]/[tool: ...]
    tag), calls load() first if the checkpoint hasn't been loaded yet.
    """
    if question.lower().startswith("web:"):
        return web_search(question[len("web:"):].strip())

    load()
    prefix, source = build_prompt(question)
    with torch.no_grad():
        with _ctx:
            start_ids = _encode(prefix)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            y = _model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    generated_ids = y[0].tolist()[len(start_ids):]
    answer = _decode(generated_ids).split("\n")[0].strip()

    call_match = CALL_RE.search(answer)
    if call_match:
        return dispatch_call(call_match)
    elif source is not None:
        return f"{answer}\n[source: {source['metadata']['title']}, rerank score: {source['score']:.2f}]"
    else:
        return f"{answer}\n[no confident source found — answered from the model's own training only]"


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
    print("model: a real live result, printed as-is, not generated by the model")
    print("(the model was never trained on web search results).")
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
