@echo off
REM Launcher for venv_data (data-only pipeline)
SET SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%venv_data\Scripts\python.exe" "%SCRIPT_DIR%run_pipeline_data.py" %*
