@echo off
REM Create virtual environment named 'venv'
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install dependencies from requirements.txt
pip install -r requirements.txt

echo.
echo Virtual environment setup complete.
pause
