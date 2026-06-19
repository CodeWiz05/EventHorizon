@echo off
echo =======================================================
echo EVENT HORIZON: Macro Regime Intelligence Engine
echo =======================================================
echo.

echo [1/3] Checking Python Virtual Environment...
:: Check if the venv already exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating 'venv'...
    python -m venv venv
    
    echo Activating venv and installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    echo Virtual environment found. Skipping dependency installation for speed.
)

echo [2/3] Booting FastAPI Backend Engine...
:: We MUST chain the activation command inside the new window so uvicorn is recognized
start cmd /k "title EventHorizon Backend API && call venv\Scripts\activate.bat && uvicorn api:app --reload"

echo [3/3] Booting React Dashboard UI...
cd frontend
if not exist "node_modules\" (
    echo First boot detected. Installing React dependencies...
    call npm install
)
start cmd /k "title EventHorizon Dashboard && npm run dev -- --open"

echo.
echo System Boot Sequence Complete.
echo Backend API running on: http://127.0.0.1:8000
echo Frontend Dashboard is launching in your default browser...
pause