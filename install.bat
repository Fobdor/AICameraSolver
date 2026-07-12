@echo off
echo ===================================================
echo  AICameraSolver Installation Script (Windows)
echo ===================================================
echo.

IF NOT EXIST "python\python.exe" (
    echo [1/4] Downloading Python 3.10.11 Embedded...
    IF NOT EXIST "python" mkdir python
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile 'python\python.zip'"
    
    echo [2/4] Extracting Python...
    powershell -Command "Expand-Archive -Path 'python\python.zip' -DestinationPath 'python\' -Force"
    del python\python.zip
    
    echo [3/4] Enabling PIP in embedded python...
    powershell -Command "(Get-Content 'python\python310._pth') -replace '#import site', 'import site' | Set-Content 'python\python310._pth'"
    
    echo [4/4] Installing PIP...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'python\get-pip.py'"
    .\python\python.exe python\get-pip.py
) ELSE (
    echo Python embedded environment already exists!
)

echo.
echo ===================================================
echo Installing AI Camera Solver Requirements...
echo ===================================================
.\python\python.exe -m pip install -r requirements.txt

echo.
echo ===================================================
echo Installation Complete! 
echo You can now run the software using launch.bat
echo ===================================================
pause
