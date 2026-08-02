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

# -----------------------------------------------------------------------------
out_dir = 'checkpoints/gpt2_local'
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
exec(open('configurator.py').read())  # overrides from command line
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"No checkpoint found at {ckpt_path} yet, nothing to talk to.")
    raise SystemExit(1)

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

load_meta = False
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

print("Type a prompt and press enter. Empty line or Ctrl+C to quit.")
print("Raw completion only, no memory of earlier turns, no retrieval, no tools.")
print()

with torch.no_grad():
    with ctx:
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt.strip():
                break
            start_ids = encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print()
