#!/bin/bash
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./install.sh first!"
    exit 1
fi
source venv/bin/activate
python3 main.py
