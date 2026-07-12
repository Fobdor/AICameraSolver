@echo off
IF NOT EXIST "python\python.exe" (
    echo Error: Python environment not found. Please run install.bat first!
    pause
    exit /b 1
)
.\python\python.exe main.py
