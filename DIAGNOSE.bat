@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python
echo Running one-frame diagnostics (about 60 seconds)...
%PY% fyp\diagnose_video.py > diagnose_output.txt 2>&1
type diagnose_output.txt
echo.
echo Saved: diagnose_output.txt + diag_*.jpg  -  send these back.
pause
