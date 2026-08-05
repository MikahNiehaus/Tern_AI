"""The one entry point: detects which phase this project is in and runs it,
auto advancing to the next phase when the current one actually finishes,
stopping (not auto relaunching) when the user interrupts one early to go
do something else. Answers the real question asked: "why do I have to
know which script to run and why doesn't it switch to the next phase on
its own" — this is that missing piece.

Six states, checked in order:
1. No base checkpoint yet -> start base pretraining from scratch.
2. Base checkpoint exists, not yet at max_iters -> resume base pretraining.
3. Base training done, Simple Wikipedia corpus not downloaded yet -> get
   it (toolstore/download_simple_wikipedia.py), so a fresh clone gets real
   paraphrase trained answers, not the copy the summary verbatim fallback,
   without anyone needing to know that script exists.
4. Simple Wikipedia downloaded, SFT dataset missing or stale -> build or
   rebuild it (quick, not itself resumable, just a one time data prep
   step). Stale means dataset_needs_rebuild() found the real inputs (either
   Wikipedia corpus, the tool traces, or build_sft_dataset.py's own logic)
   changed since the .npy files on disk were built, not just "do the files
   exist": running this again after downloading Simple Wikipedia for the
   first time, with an SFT checkpoint already fully trained on the old,
   verbatim only data, is exactly the case this exists to catch, so it
   does not just launch chat on stale weights forever.
5. SFT dataset built and current, SFT checkpoint not yet at max_iters ->
   run (or resume) SFT training. If the dataset was just rebuilt because it
   was stale, any existing SFT checkpoint was already retired (moved aside,
   never deleted) by state 4, so this starts a real fresh SFT run against
   the corrected data instead of resuming more training on top of stale
   weights.
6. Everything done -> launch toolstore/chat.py, the real RAG + tool using
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
import time

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

    toolstore_dir = os.path.join(HERE, "..", "toolstore")
    simple_wiki_tsv = os.path.join(toolstore_dir, "corpus", "simple_wikipedia_summaries", "summaries.tsv")

    # build_sft_dataset.py is a real, importable module (its only top level
    # work is fast, already installed imports: json, numpy, tiktoken, no
    # GPU or model load), not run only as a subprocess: dataset_needs_rebuild()
    # below has to be the exact same check build_sft_dataset.py's own build()
    # writes the manifest against, so there is one real implementation, not
    # two that could quietly drift apart and disagree about whether a
    # rebuild is needed.
    sys.path.insert(0, sft_dir)
    import build_sft_dataset

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

        if not os.path.exists(simple_wiki_tsv):
            print("\n=== base training done, downloading Simple Wikipedia "
                  "(for real paraphrase trained answers, not verbatim copies) ===\n")
            subprocess.run([python, "download_simple_wikipedia.py"], cwd=toolstore_dir)
            continue

        if build_sft_dataset.dataset_needs_rebuild():
            if os.path.exists(sft_ckpt):
                # The checkpoint on disk was trained against whatever
                # dataset existed before this rebuild, so resuming it with
                # --init_from=resume would keep training on top of those
                # old weights instead of a real, clean run against the
                # corrected data. Moved aside, never deleted, so sft_exists
                # below is correctly False afterward and train_sft.py's own
                # default init_from='sft_init' starts fresh from the base
                # checkpoint on the corrected dataset.
                stamp = time.strftime("%Y%m%d_%H%M%S")
                stale_dir = os.path.join(sft_dir, "checkpoints")
                stale_path = os.path.join(stale_dir, f"ckpt_stale_{stamp}.pt")
                print(f"\n=== SFT training data changed, retiring the checkpoint trained "
                      f"on the old data to {stale_path} ===\n")
                try:
                    os.replace(sft_ckpt, stale_path)
                except FileNotFoundError:
                    # The GUI's own Train tab already refuses to start a
                    # second run while one is detected running, so this is
                    # only reachable by manually launching a second
                    # orchestrate.py outside the GUI. If that happens to
                    # race this exact rename, the source is only ever
                    # missing because some other run already moved it, the
                    # same real state either way, so there is nothing left
                    # to do here, not a real failure, only printed so this
                    # is never the one silent branch in an otherwise fully
                    # narrated state machine.
                    print(f"\n=== {sft_ckpt} was already retired by another run, nothing to do here ===\n")
                except OSError as e:
                    # bad-cop measured this for real: a locked checkpoint
                    # file (chat.py still holding it via torch.load, a
                    # backup/AV tool scanning it, both realistic on a
                    # Windows box) raises PermissionError here, which used
                    # to have no handler and crash this whole days-long
                    # unattended state machine with no narration at all,
                    # per CLAUDE.md's own note that nothing here auto
                    # notifies on completion or failure. Stopping cleanly
                    # with a clear, printed reason (this file already
                    # narrates every state, this must not be the one silent
                    # exit) is safer than proceeding: continuing past a
                    # failed retirement risks building the new dataset while
                    # the old, stale checkpoint is still in place to be
                    # resumed from by mistake.
                    print(f"\n=== ERROR: could not retire stale checkpoint {sft_ckpt} "
                          f"-> {stale_path}: {e} ===\n"
                          f"=== not safe to continue: the checkpoint trained on the old "
                          f"data would still be in place to resume from by mistake. "
                          f"Check what still has {sft_ckpt} open (a running chat.py, a "
                          f"backup or antivirus scan) and rerun once it is free. ===\n")
                    return
            print("\n=== building SFT dataset ===\n")
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
        subprocess.run([python, "chat.py"], cwd=toolstore_dir)
        return


if __name__ == "__main__":
    main()
