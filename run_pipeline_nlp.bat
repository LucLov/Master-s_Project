@echo off
REM Launcher for venv_nlp (nlp-only pipeline)
SET SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%venv_nlp\Scripts\python.exe" "%SCRIPT_DIR%run_pipeline_nlp.py" %*
