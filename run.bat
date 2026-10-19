@echo off
setlocal

cd /d "%~dp0"

set "APP_DIR=%~dp0"
set "PYTHON_EXE=C:\Users\elcer\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -c "import fastapi, uvicorn, pydantic, loguru, ollama, faster_whisper, python_multipart" >nul 2>nul
if errorlevel 1 (
  echo Mancano una o piu' dipendenze nel Python selezionato.
  echo Provo a installare le dipendenze necessarie per avviare Auto Cut AI...
  "%PYTHON_EXE%" -m pip install -e .
  if errorlevel 1 (
    echo.
    echo Installazione automatica non riuscita.
    echo Esegui manualmente:
    echo "%PYTHON_EXE%" -m pip install -e .
    pause
    exit /b 1
  )
)

netstat -ano | findstr /R /C:"127.0.0.1:8000 .*LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo La porta 8000 e' gia' occupata.
  echo Chiudi l'altra finestra Auto Cut AI oppure esegui stop.bat e riprova.
  pause
  exit /b 1
)

echo Avvio server web Auto Cut AI su http://127.0.0.1:8000
start "Auto Cut AI" cmd /k ""%PYTHON_EXE%" -m uvicorn autocut_ai.app:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000/"

echo.
echo Auto Cut AI avviata.
echo App: http://127.0.0.1:8000
echo.
echo Chiudi la finestra Auto Cut AI per fermare l'app.

endlocal
