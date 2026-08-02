"""Lightweight, side effect free config values for the SFT phase. Exists
because train_sft.py is the full training script, not a separate config
file the way base training has configs/train_gpt2_local.py split out from
train.py, so runpy.run_path on train_sft.py itself would actually execute
the whole script, loading the checkpoint, loading the dataset, entering
the training loop. Found the hard way: orchestrate.py's first version did
exactly that when checking SFT status, launching real training as a side
effect of a status check. train_sft.py imports these two values as its own
out_dir/max_iters, so there is exactly one source of truth and no drift
risk; orchestrate.py reads this small file directly, never train_sft.py.
"""
out_dir = 'sft/checkpoints'
max_iters = 6000
