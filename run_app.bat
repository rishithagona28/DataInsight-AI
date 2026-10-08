@echo off
cd /d "%~dp0"
echo Starting Ana... (output is saved to run_log.txt)
echo Leave this window open while you use the app.
"..\venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 > run_log.txt 2>&1
echo.
echo Ana stopped. Last lines of run_log.txt:
type run_log.txt
pause
