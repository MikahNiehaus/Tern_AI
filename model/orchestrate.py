"""The one entry point: detects which phase this project is in and runs it,
auto advancing to the next phase when the current one actually finishes,
stopping (not auto relaunching) when the user interrupts one early to go
do something else. Answers the real question asked: "why do I have to
know which script to run and why doesn't it switch to the next phase on
its own" — this is that missing piece.

Five states, checked in order:
1. No base checkpoint yet -> start base pretraining from scratch.
2. Base checkpoint exists, not yet at max_iters -> resume base pretraining.
3. Base training done, SFT dataset not built yet -> build it (quick, not
   itself resumable, just a one time data prep step).
4. SFT dataset built, SFT checkpoint not yet at max_iters -> run (or
   resume) SFT training.
5. Everything done -> launch toolstore/chat.py, the real RAG + tool using
   chat loop (talk.py is raw base-model sampling only, useful mid training,
   not the end state once SFT is actually done).

Auto continue only fires when the checkpoint just written shows the phase
that just ran actually reached its own max_iters, confirmed via the real
checkpoint contents (iter_num), never via subprocess exit code, which
Ctrl+C and a real crash do not reliably differ on. A user pressing Ctrl+C
propagates to this process too (Windows broadcasts it to every process
sharing the console, confirmed against Microsoft's own docs, not assumed),
caught explicitly so the loop stops rather than racing the checkpoint
check.
"""
import os
import runpy
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def phase_status(config_path, ckpt_path):
    """Returns (exists, iter_num, max_iters, done)."""
    config = runpy.run_path(config_path)
    max_iters = config["max_iters"]
    if not os.path.exists(ckpt_path):
        return False, None, max_iters, False
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    iter_num = checkpoint["iter_num"]
    return True, iter_num, max_iters, iter_num >= max_iters


def run_and_check(cmd, cwd, config_path, ckpt_path, phase_name):
    """Runs cmd, waits for it, then re-checks the checkpoint. Returns True
    if the phase is now done (caller should continue the state machine),
    False if it stopped early (caller should exit, not auto relaunch)."""
    print(f"\n=== running {phase_name}: {' '.join(cmd)} ===\n")
    try:
        subprocess.run(cmd, cwd=cwd)
    except KeyboardInterrupt:
        # Ctrl+C reaches this process too, Windows broadcasts it to every
        # process sharing the console. This IS the user's interrupt, do not
        # evaluate the checkpoint to decide "did it finish naturally," it
        # did not, the user stopped it on purpose.
        print(f"\n{phase_name} interrupted. Not auto continuing. Run this again when ready.")
        return False

    _, iter_num, max_iters, done = phase_status(config_path, ckpt_path)
    if done:
        print(f"\n{phase_name} finished (iteration {iter_num} of {max_iters}), auto continuing...")
        return True
    else:
        shown = iter_num if iter_num is not None else 0
        print(f"\n{phase_name} stopped early (iteration {shown} of {max_iters}). Run this again to resume.")
        return False


def main():
    python = sys.executable
    base_config = os.path.join(HERE, "configs", "train_gpt2_local.py")
    base_ckpt = os.path.join(HERE, "checkpoints", "gpt2_local", "ckpt.pt")

    sft_dir = os.path.join(HERE, "sft")
    sft_config = os.path.join(sft_dir, "sft_config.py")
    sft_ckpt = os.path.join(sft_dir, "checkpoints", "ckpt.pt")
    sft_data_files = [
        os.path.join(sft_dir, "sft_train_x.npy"),
        os.path.join(sft_dir, "sft_train_y.npy"),
        os.path.join(sft_dir, "sft_val_x.npy"),
        os.path.join(sft_dir, "sft_val_y.npy"),
    ]

    while True:
        base_exists, base_iter, base_max, base_done = phase_status(base_config, base_ckpt)

        if not base_done:
            if not base_exists:
                cmd = [python, "train.py", "configs/train_gpt2_local.py"]
            else:
                cmd = [python, "train.py", "configs/train_gpt2_local.py", "--init_from=resume"]
            if run_and_check(cmd, HERE, base_config, base_ckpt, "base training"):
                continue
            return

        sft_data_built = all(os.path.exists(p) for p in sft_data_files)
        if not sft_data_built:
            print("\n=== base training done, building SFT dataset ===\n")
            subprocess.run([python, os.path.join("sft", "build_sft_dataset.py")], cwd=HERE)
            continue

        sft_exists, sft_iter, sft_max, sft_done = phase_status(sft_config, sft_ckpt)
        if not sft_done:
            if not sft_exists:
                cmd = [python, os.path.join("sft", "train_sft.py")]
            else:
                cmd = [python, os.path.join("sft", "train_sft.py"), "--init_from=resume"]
            if run_and_check(cmd, HERE, sft_config, sft_ckpt, "SFT training"):
                continue
            return

        print("\n" + "=" * 60)
        print("Base training and SFT both complete. Ready to chat.")
        print("=" * 60 + "\n")
        toolstore_dir = os.path.join(HERE, "..", "toolstore")
        subprocess.run([python, "chat.py"], cwd=toolstore_dir)
        return


if __name__ == "__main__":
    main()
