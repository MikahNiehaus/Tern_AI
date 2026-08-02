@echo off
set ROOT=%~dp0..
set HF_HOME=%ROOT%\.cache\huggingface
set PIP_CACHE_DIR=%ROOT%\.cache\pip
cd /d "%ROOT%\toolstore"
echo Chatting with the fine tuned model: RAG grounded, tool using.
echo Needs SFT to have finished (run.bat gets you there). Refuses to run otherwise.
echo Every response is tagged with where it came from: a source, a tool, or neither.
echo.
"%ROOT%\.venv\Scripts\python.exe" chat.py
pause
