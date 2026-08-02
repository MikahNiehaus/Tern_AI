"""Talk to whatever checkpoint currently exists, raw next token completion,
no RAG, no tools, no SFT template. This is NOT SPEC.md Part 4's chat.py,
that one needs the SFT stage to be meaningful and it has not run yet. This
is the honest, simpler thing: nanoGPT's own sample.py, one shot generation
from a fixed prompt, restructured into a loop so the model loads once and
you can type more than one prompt per session, model loading, encode,
decode all copied verbatim from sample.py, already correct and tested,
only the outer loop shape changes.

Always reads checkpoints/gpt2_local/ckpt.pt fresh, checkpoints get
overwritten in place by training, so this always talks to the latest one,
no extra "find the latest checkpoint" logic needed. Quality depends
entirely on how far training has actually gotten, early on this will be
close to gibberish, that is expected, not a bug, see SPEC.md's own honest
acceptance criteria.

load()/answer() split from the CLI loop so gui.py can import this module
and call into it directly, worker thread, no subprocess, no stdin piping
(swiper researched this: piping to an already-running interactive input()
loop is a documented fragile pattern on Windows, no pty). load() is
idempotent (safe to call more than once) and saves/restores sys.argv
around configurator.py's exec, so a GUI's own argv never leaks in as a
bogus config override (found by swiper: configurator.py reads sys.argv[1:]
unconditionally at import/exec time).
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# Windows' console default codepage (cp1252) mangles the curly quotes and
# dashes tiktoken's gpt2 vocab decodes into (seen for real: apostrophes
# printed as U+FFFD). Force stdout to UTF-8 so real text prints correctly.
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
out_dir = os.path.join(HERE, 'checkpoints', 'gpt2_local')
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# -----------------------------------------------------------------------------

_model = None
_encode = None
_decode = None
_ctx = None


def is_loaded():
    """True once load() has actually loaded a checkpoint. A caller (gui.py)
    that wants to force device before the first load must check this
    first: the model, once loaded, stays on whatever device it loaded onto,
    reassigning the module level device global afterward does not move it,
    and answer() would crash on the next call with a real device mismatch
    between the model and a freshly built input tensor. Found by bad-cop,
    reproduced for real.
    """
    return _model is not None


def load():
    """Loads the latest checkpoint. Safe to call more than once, only does
    real work the first time. Raises FileNotFoundError if no checkpoint
    exists yet, callers decide how to show that (CLI exits, GUI shows a
    message), this function never calls sys.exit itself so it stays
    import-safe.
    """
    global _model, _encode, _decode, _ctx
    global out_dir, max_new_tokens, temperature, top_k, seed, device, dtype
    if _model is not None:
        return

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        exec(open(os.path.join(HERE, 'configurator.py')).read())  # overrides from command line
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
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path} yet, nothing to talk to.")

    print(f"Loading {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"(trained to iteration {checkpoint['iter_num']})")
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

    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join(HERE, 'data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        _encode = lambda s: [stoi[c] for c in s]
        _decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        _encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        _decode = lambda l: enc.decode(l)


def answer(prompt):
    """Raw completion for prompt, returns the decoded continuation. Calls
    load() first if the checkpoint hasn't been loaded yet.
    """
    load()
    with torch.no_grad():
        with _ctx:
            start_ids = _encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            y = _model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            return _decode(y[0].tolist())


if __name__ == "__main__":
    try:
        load()
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    print("Type a prompt and press enter. Empty line or Ctrl+C to quit.")
    print("Raw completion only, no memory of earlier turns, no retrieval, no tools.")
    print()

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        print(answer(prompt))
        print()
