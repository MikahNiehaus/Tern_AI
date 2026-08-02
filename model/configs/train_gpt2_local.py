# GPT-2 small (124M), scaled to run on one RTX 4070 SUPER (12GB VRAM), locally.
#
# Architecture is unchanged from nanoGPT's own config/train_gpt2.py (still
# real GPT-2 small: 12 layers, 12 heads, 768 embd, 1024 block_size). All
# field names here were confirmed against train.py directly (configurator.py
# exec()'s this file into train.py's globals, so a typo'd name would
# silently do nothing, not error, worth checking for exactly that reason).
#
# batch_size, gradient_accumulation_steps, max_iters, and compile were
# changed from nanoGPT's original, and every one of those numbers was
# measured on this exact GPU, not guessed:
#
#   batch_size=6, not the originally planned 8. Found the hard way, running
#   the real script for real (not the isolated synthetic benchmark that
#   originally justified 8): actual train.py, with GradScaler, gradient
#   clipping, and the full real state resident, measured 11.8GB used of
#   this card's 12.28GB actual usable total at batch_size=8, only 446MB of
#   headroom. That is enough to trigger a real, documented Windows NVIDIA
#   driver behavior (driver 536.40+, since June 2023): CUDA allocations
#   silently spill into system RAM instead of raising an out of memory
#   error when dedicated VRAM is nearly exhausted, and PyTorch cannot tell
#   the difference, so nothing crashes, it just gets much slower. Measured
#   effect: 39 seconds per iteration at batch_size=8, mfu about 2.7%, high
#   reported GPU "utilization" but low power draw, the exact signature of
#   waiting on PCIe transfers to and from system RAM, not real compute.
#   batch_size=6 dropped real VRAM use to 9.97GB, a real 2.3GB of headroom,
#   and iteration time to about 7.96 to 7.99 seconds, confirmed stable
#   across multiple iterations, not a fluke, mfu rose to about 13.5%. The
#   earlier isolated benchmark that said batch_size=8 was safe at 9.49GB
#   was real but incomplete, it never included GradScaler, gradient
#   clipping, or a full training loop's actual peak memory pattern, only
#   forward+backward+optimizer.step() in isolation.
#
#   gradient_accumulation_steps=64 (was 48) keeps the same effective tokens
#   per optimizer step (6 * 1024 * 64 = 393,216, identical to the original
#   8 * 1024 * 48 plan) so training dynamics are unchanged by the batch
#   size correction, only the memory footprint and wall clock speed are.
#
#   Measured throughput at the corrected settings: about 7.97s/iteration,
#   close to the original isolated estimate (8.7s) and noticeably better
#   than the mistaken batch_size=8 in-sysmem-fallback number (39s). At this
#   real rate, max_iters=25000 fits inside the 3 day base training window
#   with real margin, matching what was originally planned before the VRAM
#   pressure bug was found and fixed.
#
#   compile=False because torch.compile was tested directly on this machine
#   and failed (TritonMissing: no working Triton install on native Windows),
#   not assumed from a general claim about Windows support.
#
#   warmup_iters=1000 is a deliberate choice, not nanoGPT's own default of
#   2000: scaling that default proportionally to a 25,000 iter run would
#   give an unreasonably short 80 step warmup. 1000 (4 percent of this
#   shorter run) follows standard practice of a larger warmup fraction for
#   shorter runs, since Adam's variance estimates need a real number of
#   absolute steps to stabilize regardless of total run length.
#
#   eval_interval=50 and eval_iters=50, not the shakespeare_char config's
#   original 250/200 that was left in place through the first pass. Found
#   the hard way, running train.bat for real: at this config's measured
#   8.7s/iteration, eval_interval=250 meant about 36 minutes before the
#   first checkpoint, since nanoGPT's own checkpoint code has an
#   if iter_num > 0 guard, nothing saves at iteration 0 even with
#   always_save_checkpoint=True. That directly undermines the pause and
#   resume design, a pause could lose up to 36 minutes. 50 iterations is
#   about 7 minutes between checkpoints instead, eval_iters lowered to
#   match so the eval pass itself does not become the new overhead.

out_dir = 'checkpoints/gpt2_local'
eval_interval = 50 # measured: 250 meant ~36min between checkpoints, too long for pause and resume
eval_iters = 50
log_interval = 10

always_save_checkpoint = True # required for the pause/resume workflow

wandb_log = False # fully local project, no external dependency

dataset = 'openwebtext'
gradient_accumulation_steps = 64
batch_size = 6
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 25000
lr_decay_iters = 25000
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
warmup_iters = 1000

device = 'cuda'
compile = False # measured: torch.compile fails on this machine, Triton not available on native Windows
