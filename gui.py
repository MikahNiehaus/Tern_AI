"""Single GUI entry point, replaces run.bat/stop.bat/talk.bat/chat.bat.
Three tabs: Train (start/stop/watch pretraining and SFT), Talk (raw
completion), Chat (RAG + tool use).

Location independent the same way the batch files were: __file__-relative
paths, not hardcoded to any one drive. Sets HF_HOME/PIP_CACHE_DIR the same
way the batch files did, so downloads still land under this project instead
of the OS user profile.

Train tab launches orchestrate.py as a real subprocess (stdout piped,
streamed into the log view via a background thread and a queue.Queue,
drained on the Tk main loop with root.after — never touching a Tk widget
from the worker thread directly, and never calling thread.join() from a Tk
callback, both are documented ways to freeze a Tkinter GUI). It reuses the
exact PowerShell process detection scripts/run.bat and stop.bat already
used and proved correct (an earlier window-title based check was wrong,
CUDA startup resets the console title), so "is training already running"
means the same thing here it always did. If training was started outside
this GUI (true on this machine as of writing this file), the GUI can only
detect that and offer Stop, it was never given that process's stdout
handle and does not attempt to intercept it after the fact.

Talk and Chat tabs import model/talk.py and toolstore/chat.py directly and
call their load()/answer()/answer_question() functions from a worker
thread (same queue+after pattern), rather than spawning them as
subprocesses: piping to an already-interactive input() loop is a
documented fragile pattern on Windows (no pty, buffering deadlocks), and
both files were already split into an importable load()/answer() shape
for exactly this. If training is detected running, both tabs force
device='cpu' before their first load(), since pretraining already uses
most of the card's 12GB VRAM by design (CLAUDE.md) and loading a second
model onto the GPU at the same time is exactly the risk that rule exists
to prevent.
"""
import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gui")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TOOLSTORE_DIR = os.path.join(HERE, "toolstore")
VENV_PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")

os.environ["HF_HOME"] = os.path.join(HERE, ".cache", "huggingface")
os.environ["PIP_CACHE_DIR"] = os.path.join(HERE, ".cache", "pip")

sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, TOOLSTORE_DIR)

CREATE_NO_WINDOW = 0x08000000

# Verbatim from scripts/run.bat/stop.bat, the exact filter already proven
# correct on this machine, only reshaped to print a parseable marker
# instead of exiting zero or one (run.bat/stop.bat were shell scripts that
# only needed an exit code, this needs the PID too).
_FILTER = (
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "Where-Object { $_.CommandLine -like '*train_gpt2_local.py*' -or $_.CommandLine -like '*train_sft.py*' }"
)
CHECK_RUNNING_PS = (
    f"$procs = {_FILTER}; "
    "if ($procs) { Write-Host ('RUNNING ' + $procs[0].ProcessId) } else { Write-Host 'NOTRUNNING' }"
)
STOP_PS = (
    f"$procs = {_FILTER}; "
    "if ($procs) { $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; "
    "Write-Host 'Stopped, VRAM freed.' } else { Write-Host 'Nothing was running.' }"
)


def check_training_running():
    """Read-only. Returns a PID string if a train_gpt2_local.py/
    train_sft.py process is running (started by this GUI or not), None
    otherwise. Never touches the process.
    """
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", CHECK_RUNNING_PS],
        capture_output=True, text=True, creationflags=CREATE_NO_WINDOW,
    )
    out = result.stdout.strip()
    if out.startswith("RUNNING"):
        return out.split(" ", 1)[1]
    return None


class TrainTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.proc = None  # only set when THIS gui launched it
        self.log_queue = queue.Queue()

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)
        self.status_var = tk.StringVar(value="Checking...")
        ttk.Label(top, textvariable=self.status_var, wraplength=520).pack(side="left", fill="x", expand=True)
        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.start_btn.pack(side="right", padx=4)
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop)
        self.stop_btn.pack(side="right", padx=4)

        self.log = scrolledtext.ScrolledText(self, state="disabled", height=30)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

        self.refresh_status()
        self.after(200, self.drain_queue)
        self.after(4000, self.periodic_status_check)

    def append_log(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line)
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh_status(self):
        launched_here = self.proc is not None and self.proc.poll() is None
        if launched_here:
            self.status_var.set(f"Running (started by this GUI, PID {self.proc.pid}).")
            self.start_btn.state(["disabled"])
            self.stop_btn.state(["!disabled"])
            return
        pid = check_training_running()
        if pid is not None:
            self.status_var.set(
                f"Already running (PID {pid}), started outside this GUI. "
                "Live output isn't visible here, Stop still works."
            )
            self.start_btn.state(["disabled"])
            self.stop_btn.state(["!disabled"])
        else:
            self.status_var.set("Not running. No checkpoint yet starts base training, mid checkpoint resumes it.")
            self.start_btn.state(["!disabled"])
            self.stop_btn.state(["disabled"])

    def periodic_status_check(self):
        self.refresh_status()
        self.after(4000, self.periodic_status_check)

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        if check_training_running() is not None:
            self.append_log("Already running outside this GUI, refusing to start a second one.\n")
            self.refresh_status()
            return
        self.proc = subprocess.Popen(
            [VENV_PY, "orchestrate.py"],
            cwd=MODEL_DIR, env=dict(os.environ),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._reader_thread, daemon=True).start()
        self.refresh_status()

    def _reader_thread(self):
        for line in self.proc.stdout:
            self.log_queue.put(line)
        self.log_queue.put("[process exited]\n")

    def drain_queue(self):
        try:
            while True:
                self.append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(200, self.drain_queue)

    def stop(self):
        subprocess.run(["powershell", "-NoProfile", "-Command", STOP_PS],
                        capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
        self.append_log("Stop requested.\n")
        self.after(500, self.refresh_status)


class _ChatLikeTab(ttk.Frame):
    """Shared shape for Talk and Chat: a scrolled output view, an entry box,
    a Send button, worker thread calls into the target module, result comes
    back through a queue. Subclasses set self.module_name and implement
    _call(module, text).
    """
    module_name = None
    hint_text = None

    def __init__(self, master):
        super().__init__(master)
        self.result_queue = queue.Queue()
        self._module = None
        self._warned_cpu = False
        self._busy = False

        self.output = scrolledtext.ScrolledText(self, state="disabled", height=25)
        self.output.pack(fill="both", expand=True, padx=8, pady=8)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(0, 4))
        self.entry = ttk.Entry(bottom)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.send())
        self.send_btn = ttk.Button(bottom, text="Send", command=self.send)
        self.send_btn.pack(side="right", padx=4)

        if self.hint_text:
            ttk.Label(self, text=self.hint_text, wraplength=760).pack(padx=8, pady=(0, 8), anchor="w")

        self.after(100, self.drain_queue)

    def append(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def send(self):
        # Found by bad-cop, real execution: the <Return> binding calls this
        # directly, bypassing the disabled Send button, so two quick Enters
        # (or Enter then a click) could start two worker threads that race
        # over the shared module object and its device global. This flag is
        # the actual guard, the button's disabled state is only a visual
        # hint, not the thing that stops a second thread from starting.
        if self._busy:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self._busy = True
        self.entry.delete(0, "end")
        self.append(f"> {text}\n")
        self.send_btn.state(["disabled"])
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _worker(self, text):
        import importlib
        module = self._module or importlib.import_module(self.module_name)
        self._module = module
        # Base pretraining already uses most of this card's VRAM by design
        # (CLAUDE.md); loading a second model onto the GPU at the same time
        # is exactly the thing that rule exists to prevent, so force CPU
        # for this module if training is live right now. Only before the
        # first load though: found by bad-cop, real execution, that
        # reassigning module.device after a model has already loaded onto
        # a device does not move it, load() is idempotent and answer()
        # would crash on the very next call with a real device mismatch
        # between the resident model and a freshly built input tensor.
        if not module.is_loaded() and check_training_running() is not None:
            module.device = 'cpu'
            if not self._warned_cpu:
                self.result_queue.put(("note", "[training is running, using CPU for this session, slower but safe]"))
                self._warned_cpu = True
        try:
            result = self._call(module, text)
        except Exception as e:
            logger.error("chat turn failed (%s): %s", self.module_name, e, exc_info=True)
            result = f"[error: {e}]"
        self.result_queue.put(("answer", result))

    def _call(self, module, text):
        raise NotImplementedError

    def drain_queue(self):
        try:
            while True:
                kind, text = self.result_queue.get_nowait()
                self.append(text + "\n\n")
                if kind == "answer":
                    self.send_btn.state(["!disabled"])
                    self._busy = False
        except queue.Empty:
            pass
        self.after(100, self.drain_queue)


class TalkTab(_ChatLikeTab):
    module_name = "talk"
    hint_text = "Raw completion only, no retrieval, no tools, no memory of earlier turns."

    def _call(self, module, text):
        return module.answer(text)


class ChatTab(_ChatLikeTab):
    module_name = "chat"
    hint_text = "Needs SFT to have finished. Prefix a question with 'web:' to search DuckDuckGo live instead of the local model."

    def _call(self, module, text):
        return module.answer_question(text)


def main():
    root = tk.Tk()
    root.title("Tern AI")
    root.geometry("820x640")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    notebook.add(TrainTab(notebook), text="Train")
    notebook.add(TalkTab(notebook), text="Talk")
    notebook.add(ChatTab(notebook), text="Chat")

    root.mainloop()


if __name__ == "__main__":
    main()
