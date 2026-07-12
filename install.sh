#!/bin/bash
echo "==================================================="
echo " AICameraSolver Installation Script (Mac/Linux)"
echo "==================================================="
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 could not be found."
    echo "Please install Python 3.10+ using your package manager (apt/brew) or from python.org"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "[1/2] Creating virtual environment (venv)..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

echo "[2/2] Installing AI Camera Solver Requirements..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "==================================================="
echo "Installation Complete!"
echo "You can now run the software using ./launch.sh"
echo "==================================================="
