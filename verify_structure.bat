@echo off
echo Verifying output structure with correct Python interpreter...
cd "%~dp0"
"C:\Users\nicki\Documents\Cline\venv\Scripts\python.exe" python_wrapper.py verify %*
