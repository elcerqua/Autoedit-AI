@echo off
setlocal

echo Cerco processi in ascolto su 127.0.0.1:8000...

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:"127.0.0.1:8000 .*LISTENING"') do (
  echo Trovato processo PID %%p sulla porta 8000.
  choice /M "Vuoi fermarlo"
  if errorlevel 2 (
    echo Operazione annullata.
    exit /b 1
  )
  taskkill /PID %%p /F
  if errorlevel 1 (
    echo Impossibile fermare il processo %%p.
    exit /b 1
  )
  echo Processo %%p fermato.
)

echo Porta 8000 libera oppure nessun processo in ascolto trovato.
endlocal
