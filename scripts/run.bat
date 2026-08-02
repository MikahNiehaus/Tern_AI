@echo off
title GPT2-Training
set ROOT=%~dp0..
set HF_HOME=%ROOT%\.cache\huggingface
set PIP_CACHE_DIR=%ROOT%\.cache\pip
cd /d "%ROOT%\model"

powershell -NoProfile -Command "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*train_gpt2_local.py*' -or $_.CommandLine -like '*train_sft.py*' }; if ($procs) { Write-Host 'Already running (PID' $procs[0].ProcessId '). Run stop.bat first.'; exit 1 }"
if errorlevel 1 (
    pause
    exit /b 1
)

echo One button: does the right thing automatically, whatever state you left off in.
echo No checkpoint yet = starts base training. Mid base training = resumes it.
echo Base done = builds the SFT dataset and trains it. SFT done = lets you talk to it.
echo Close this window or press Ctrl+C any time to pause and go do something else,
echo run.bat again later picks up exactly where you left off, on the same phase you were on.
echo When a phase actually finishes (not just paused), it moves to the next one on its own.
echo.
"%ROOT%\.venv\Scripts\python.exe" orchestrate.py
echo.
pause
