@echo off
set ROOT=%~dp0..
set HF_HOME=%ROOT%\.cache\huggingface
set PIP_CACHE_DIR=%ROOT%\.cache\pip
cd /d "%ROOT%\model"
echo Talking to the latest checkpoint, raw completion only.
echo No RAG, no tools, no memory of earlier turns yet, that needs the SFT stage.
echo Quality depends on how far training has gotten so far, expect gibberish early on.
echo.
"%ROOT%\.venv\Scripts\python.exe" talk.py
pause
