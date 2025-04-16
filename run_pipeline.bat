@echo off
echo Running pipeline with correct Python interpreter...
cd "%~dp0"
"C:\Users\nicki\Documents\Cline\venv\Scripts\python.exe" python_wrapper.py pipeline %*
