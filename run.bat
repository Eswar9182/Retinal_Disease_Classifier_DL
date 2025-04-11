@echo off
cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing required packages...
pip install -r requirements.txt

echo Starting Flask app and opening browser...
start http://127.0.0.1:5000
python app.py

pause
